"""
ner_trainer.py — Fine-tuned NER trainer for process-safety domain.

Problem with generic BERT NER:
  dslim/bert-base-NER was trained on news text (CoNLL-2003).
  It has never seen "TAHH-201 trips the ESD on high reactor temperature"
  and will never correctly label TAHH-201 as an instrument or ESD as a safeguard.

This module:
  1. Generates a silver-standard training corpus from the CSB sample reports
     using the existing rule-based NER as a weak supervisor
  2. Fine-tunes a BERT token classifier on the safety-domain labels:
     CHEM, EQUIP, PARAM, CAUSE, CONSEQ, SAFEGUARD, LOC, ORG
  3. Saves the fine-tuned model to models/ner_finetuned/
  4. Provides a drop-in replacement for BertNERLayer

Fine-tuning data sources:
  - Auto-annotated CSB sample reports (weak supervision)
  - Manually annotated sentences (gold standard, if provided)
  - Augmented sentences from KB cause/consequence text
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional
from loguru import logger

from config import MODEL_DIR, PROCESSED_DIR, SAMPLE_DIR

FINETUNED_MODEL_DIR = MODEL_DIR / "ner_finetuned"

# ── Label scheme (IOB2) ───────────────────────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-CHEM",    "I-CHEM",
    "B-EQUIP",   "I-EQUIP",
    "B-PARAM",   "I-PARAM",
    "B-CAUSE",   "I-CAUSE",
    "B-CONSEQ",  "I-CONSEQ",
    "B-SAFEGUARD","I-SAFEGUARD",
    "B-LOC",     "I-LOC",
    "B-ORG",     "I-ORG",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Silver corpus generation (weak supervision)
# ══════════════════════════════════════════════════════════════════════════════

def generate_silver_corpus(
    text_files: Optional[list[Path]] = None,
    output_path: Optional[Path] = None,
    max_sentences: int = 5000,
) -> list[dict]:
    """
    Auto-annotate sentences using the rule-based NER layer.
    Produces IOB2-tagged token sequences for fine-tuning.

    Returns list of {"tokens": [...], "ner_tags": [...]} dicts.
    """
    from src.ner_pipeline import ChemSafetyNERLayer
    from src.ingestion    import ingest_text, segment_sentences

    rule_ner = ChemSafetyNERLayer()

    # Collect source text
    sources: list[str] = []
    if text_files:
        for f in text_files:
            sources.append(f.read_text(encoding="utf-8", errors="replace"))
    else:
        # Use sample reports + KB text
        for f in SAMPLE_DIR.glob("*.txt"):
            sources.append(f.read_text(encoding="utf-8", errors="replace"))
        sources.extend(_generate_kb_sentences())

    all_examples: list[dict] = []

    for source_text in sources:
        sentences = segment_sentences(source_text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue

            tokens = _simple_tokenise(sent)
            if not tokens:
                continue

            # Get entity spans from rule NER
            entities = rule_ner.extract(sent)

            # Build IOB2 tag sequence
            tags = _span_to_iob2(sent, tokens, entities)
            all_examples.append({"tokens": tokens, "ner_tags": tags})

            if len(all_examples) >= max_sentences:
                break
        if len(all_examples) >= max_sentences:
            break

    # Shuffle
    random.shuffle(all_examples)

    logger.info(f"Silver corpus: {len(all_examples)} annotated sentences")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_examples, fh, indent=2)
        logger.info(f"Saved silver corpus → {output_path}")

    return all_examples


def _simple_tokenise(text: str) -> list[str]:
    """Whitespace + punctuation tokeniser matching BERT subword boundaries."""
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return [t for t in tokens if t.strip()]


def _span_to_iob2(text: str, tokens: list[str], entities) -> list[str]:
    """Convert character-span entities to IOB2 token tags."""
    tags = ["O"] * len(tokens)

    # Build token character offsets
    token_spans = []
    cursor = 0
    for tok in tokens:
        start = text.find(tok, cursor)
        if start == -1:
            token_spans.append((cursor, cursor + len(tok)))
        else:
            token_spans.append((start, start + len(tok)))
            cursor = start + len(tok)

    for ent in entities:
        ent_start = ent.start if hasattr(ent, "start") else ent.get("start", 0)
        ent_end   = ent.end   if hasattr(ent, "end")   else ent.get("end", 0)
        label     = ent.label if hasattr(ent, "label") else ent.get("label", "O")

        first = True
        for i, (ts, te) in enumerate(token_spans):
            # Token overlaps with entity span
            if ts < ent_end and te > ent_start:
                prefix = "B-" if first else "I-"
                iob_label = f"{prefix}{label}"
                if iob_label in LABEL2ID:
                    tags[i] = iob_label
                    first = False

    return tags


def _generate_kb_sentences() -> list[str]:
    """Generate training sentences from the HAZOP deviation KB."""
    from src.hazop_engine import DEVIATION_KB
    sentences = []
    for (gw, param), entry in DEVIATION_KB.items():
        for cause in entry.get("causes", []):
            sentences.append(cause)
        for conseq in entry.get("consequences", []):
            sentences.append(conseq)
        for sg in entry.get("safeguards", []) + entry.get("recommended", []):
            sentences.append(sg)
    return sentences


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

FINETUNE_BASE_MODEL = "dslim/bert-base-NER"


def finetune(
    corpus: list[dict],
    output_dir: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.1,
) -> Path:
    """
    Fine-tune BERT on the silver corpus.
    Requires: transformers, torch, datasets

    Returns path to saved model directory.
    """
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification,
        )
        import torch
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            f"Fine-tuning requires transformers + torch + datasets: {e}"
        )

    output_dir = Path(output_dir) if output_dir else FINETUNED_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {FINETUNE_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_BASE_MODEL)
    model     = AutoModelForTokenClassification.from_pretrained(
        FINETUNE_BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Split train/val
    random.shuffle(corpus)
    split = int(len(corpus) * (1 - val_split))
    train_data = corpus[:split]
    val_data   = corpus[split:]

    train_ds = Dataset.from_list(train_data)
    val_ds   = Dataset.from_list(val_data)

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )
        all_labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(LABEL2ID.get(tags[word_id], 0))
                else:
                    # Continuation subword — use I- label
                    tag = tags[word_id]
                    if tag.startswith("B-"):
                        tag = "I-" + tag[2:]
                    label_ids.append(LABEL2ID.get(tag, 0))
                prev_word_id = word_id
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    train_ds = train_ds.map(tokenize_and_align, batched=True)
    val_ds   = val_ds.map(tokenize_and_align,   batched=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )

    collator = DataCollatorForTokenClassification(tokenizer)
    trainer  = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting fine-tuning…")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label config
    with open(output_dir / "label_config.json", "w") as fh:
        json.dump({"label_list": LABEL_LIST, "id2label": ID2LABEL,
                   "label2id": LABEL2ID}, fh, indent=2)

    logger.info(f"Fine-tuned model saved → {output_dir}")
    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Fine-tuned NER layer (drop-in replacement)
# ══════════════════════════════════════════════════════════════════════════════

class FinetunedNERLayer:
    """
    Drop-in replacement for BertNERLayer using the fine-tuned safety model.
    Falls back to generic BERT if fine-tuned model is not available.
    """

    def __init__(self, model_dir: Optional[Path] = None, device: int = -1):
        model_dir = Path(model_dir) if model_dir else FINETUNED_MODEL_DIR

        try:
            from transformers import pipeline as hf_pipeline
            if not model_dir.exists():
                raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}")
            self._pipe = hf_pipeline(
                "ner",
                model=str(model_dir),
                tokenizer=str(model_dir),
                aggregation_strategy="simple",
                device=device,
            )
            self._source = "finetuned"
            logger.info(f"Fine-tuned NER loaded from {model_dir}")
        except Exception as exc:
            logger.warning(f"Fine-tuned model unavailable ({exc}) — using base BERT")
            from transformers import pipeline as hf_pipeline
            from config import NER_MODEL_NAME
            self._pipe = hf_pipeline(
                "ner",
                model=NER_MODEL_NAME,
                aggregation_strategy="simple",
                device=device,
            )
            self._source = "bert_base"

    def extract(self, text: str, sentence: str = "") -> list:
        from src.ner_pipeline import Entity
        from config import ENTITY_TYPES
        entities = []
        try:
            raw = self._pipe(text[:512])
        except Exception as exc:
            logger.warning(f"NER inference error: {exc}")
            return []
        for item in raw:
            label = item.get("entity_group", "O").replace("B-", "").replace("I-", "")
            if label == "O":
                continue
            entities.append(Entity(
                text=item["word"].strip(),
                label=label,
                label_desc=ENTITY_TYPES.get(label, label),
                start=item["start"],
                end=item["end"],
                score=round(float(item.get("score", 0.9)), 4),
                source=self._source,
                sentence=sentence,
            ))
        return entities


# ══════════════════════════════════════════════════════════════════════════════
# CLI — run fine-tuning from terminal
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune NER for HAZOP safety domain")
    parser.add_argument("--epochs",    type=int,   default=3)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--max-sents", type=int,   default=5000)
    parser.add_argument("--output",    type=str,   default=str(FINETUNED_MODEL_DIR))
    args = parser.parse_args()

    print("Generating silver corpus…")
    corpus = generate_silver_corpus(max_sentences=args.max_sents)
    print(f"Corpus: {len(corpus)} sentences")

    print("Fine-tuning BERT…")
    out = finetune(corpus, output_dir=Path(args.output), epochs=args.epochs,
                   batch_size=args.batch)
    print(f"Done → {out}")
