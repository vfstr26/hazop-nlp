"""
ner_pipeline.py — Named Entity Recognition pipeline for HAZOP NLP system.

Two-layer NER strategy:
  Layer 1 — General NER (dslim/bert-base-NER via HuggingFace Transformers)
             Extracts: PER, ORG, LOC, MISC
  Layer 2 — Chemical / Process-Safety NER (rule-based + pattern matching)
             Extracts: CHEM, EQUIP, PARAM, CAUSE, CONSEQ, SAFEGUARD

The two layers are merged and deduplicated into a unified entity list
that downstream modules (HAZOP engine, LLM) consume.

Classes:
  Entity           — dataclass for a single extracted entity
  BertNERLayer     — wraps HuggingFace pipeline for general NER
  ChemSafetyNERLayer — regex/gazeteer NER for process-safety concepts
  HAZOPNERPipeline — orchestrates both layers + post-processing
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from loguru import logger

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification
    _TRANSFORMERS = True
except ImportError:
    _TRANSFORMERS = False
    logger.warning("transformers not installed — BERT NER layer disabled.")

try:
    import spacy
    _SPACY = True
except ImportError:
    _SPACY = False

from config import NER_MODEL_NAME, NER_MAX_LENGTH, ENTITY_TYPES, MODEL_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Entity Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text: str
    label: str           # e.g. CHEM, EQUIP, CAUSE …
    label_desc: str      # human-readable label description
    start: int           # character offset in source text
    end: int
    score: float = 1.0   # confidence (0–1)
    source: str = ""     # "bert" | "rule" | "spacy"
    sentence: str = ""   # the sentence this entity came from

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Gazetteers & Patterns
# ══════════════════════════════════════════════════════════════════════════════

# --- Chemicals & hazardous substances ----------------------------------------
CHEMICAL_TERMS = [
    # Flammable gases & vapours
    "hydrogen", "hydrogen sulfide", "hydrogen sulphide", "ammonia", "methane",
    "propane", "butane", "ethylene", "acetylene", "natural gas", "LPG",
    "ethanol", "methanol", "isopropanol", "benzene", "toluene", "xylene",
    "naphtha", "gasoline", "petrol", "diesel", "fuel oil", "crude oil",
    # Toxic / reactive
    "chlorine", "phosgene", "hydrochloric acid", "hydrofluoric acid",
    "sulfuric acid", "sulphuric acid", "nitric acid", "caustic soda",
    "sodium hydroxide", "hydrogen peroxide", "chlorine dioxide",
    "acrylonitrile", "vinyl chloride", "ethylene oxide", "propylene oxide",
    # Cryogenics / asphyxiants
    "liquid nitrogen", "liquid oxygen", "carbon dioxide", "argon",
    # Dusts
    "ammonium nitrate", "potassium nitrate", "aluminium dust", "coal dust",
    "grain dust", "wood dust",
]

# --- Process equipment -------------------------------------------------------
EQUIPMENT_TERMS = [
    "reactor", "distillation column", "column", "vessel", "tank",
    "storage tank", "pressure vessel", "heat exchanger", "condenser",
    "reboiler", "pump", "compressor", "valve", "relief valve",
    "safety valve", "pressure relief valve", "PRV", "PSV",
    "rupture disc", "flare", "scrubber", "absorber", "pipeline",
    "pipe", "piping", "fitting", "flange", "nozzle", "manway",
    "agitator", "mixer", "centrifuge", "filter", "dryer", "furnace",
    "fired heater", "boiler", "autoclave", "silo", "hopper", "conveyor",
    "blower", "fan", "turbine", "separator", "drum", "knockout drum",
]

# --- Process parameters ------------------------------------------------------
PARAMETER_TERMS = [
    "temperature", "pressure", "flow", "flow rate", "level", "concentration",
    "pH", "viscosity", "density", "humidity", "speed", "voltage", "current",
    "power", "heat", "cooling", "heating", "reaction rate", "conversion",
    "selectivity", "yield", "composition", "vapor pressure",
]

# --- Initiating causes -------------------------------------------------------
CAUSE_TERMS = [
    "leak", "leakage", "rupture", "overpressure", "over-pressure",
    "overfilling", "over-filling", "blockage", "plugging", "clogging",
    "corrosion", "erosion", "fatigue", "cracking", "brittle fracture",
    "thermal shock", "runaway", "runaway reaction", "exothermic reaction",
    "loss of containment", "loss of cooling", "loss of power",
    "instrument failure", "control failure", "valve failure",
    "pump failure", "human error", "operator error", "ignition",
    "static electricity", "hot work", "maintenance error",
    "design error", "wrong chemical", "contamination",
    "reverse flow", "backflow", "two-phase flow", "water hammer",
]

# --- Consequences ------------------------------------------------------------
CONSEQUENCE_TERMS = [
    "explosion", "fire", "flash fire", "jet fire", "pool fire",
    "BLEVE", "boilover", "toxic release", "toxic cloud", "toxic plume",
    "vapor cloud explosion", "VCE", "dust explosion",
    "fatality", "fatalities", "injury", "injuries", "death", "deaths",
    "burn", "burns", "asphyxiation", "poisoning", "environmental damage",
    "spill", "release", "overpressure wave", "blast wave",
    "structural damage", "equipment damage", "plant shutdown",
]

# --- Safeguards / barriers ---------------------------------------------------
SAFEGUARD_TERMS = [
    "pressure relief", "safety valve", "PRV", "PSV", "rupture disc",
    "interlock", "SIS", "safety instrumented system", "ESD", "ESD valve",
    "emergency shutdown", "BPCS", "DCS", "alarm", "high-level alarm",
    "high-pressure alarm", "high-temperature alarm", "trip",
    "fire suppression", "sprinkler", "deluge", "foam system",
    "gas detector", "toxic gas detector", "flammable gas detector",
    "flame detector", "firewall", "bund", "bunding", "dike",
    "containment", "secondary containment", "grounding", "bonding",
    "ventilation", "purge", "inert gas blanket", "nitrogen blanket",
    "permit to work", "PTW", "LOTO", "lockout tagout",
    "operator training", "procedure", "SOP", "PPE",
    "personal protective equipment", "HAZOP", "LOPA", "QRA",
    "double block and bleed", "check valve", "non-return valve",
]


def _build_pattern(terms: list[str]) -> re.Pattern:
    """
    Compile a case-insensitive whole-word pattern from a list of terms.
    Longer terms first to avoid partial matches.
    """
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)


_CHEM_PATTERN   = _build_pattern(CHEMICAL_TERMS)
_EQUIP_PATTERN  = _build_pattern(EQUIPMENT_TERMS)
_PARAM_PATTERN  = _build_pattern(PARAMETER_TERMS)
_CAUSE_PATTERN  = _build_pattern(CAUSE_TERMS)
_CONSEQ_PATTERN = _build_pattern(CONSEQUENCE_TERMS)
_SAFE_PATTERN   = _build_pattern(SAFEGUARD_TERMS)

_RULE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_CHEM_PATTERN,   "CHEM"),
    (_EQUIP_PATTERN,  "EQUIP"),
    (_PARAM_PATTERN,  "PARAM"),
    (_CAUSE_PATTERN,  "CAUSE"),
    (_CONSEQ_PATTERN, "CONSEQ"),
    (_SAFE_PATTERN,   "SAFEGUARD"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: BERT NER
# ══════════════════════════════════════════════════════════════════════════════

class BertNERLayer:
    """
    Wraps HuggingFace token-classification pipeline (dslim/bert-base-NER).
    Handles long documents by chunking into MAX_LENGTH-token windows.
    """

    # Map BERT output labels to our unified label set
    LABEL_MAP = {
        "B-ORG": "ORG",  "I-ORG": "ORG",
        "B-PER": "PERSON", "I-PER": "PERSON",
        "B-LOC": "LOC",  "I-LOC": "LOC",
        "B-MISC": "MISC", "I-MISC": "MISC",
    }

    def __init__(self, model_name: str = NER_MODEL_NAME, device: int = -1):
        """
        device: -1 = CPU, 0 = first GPU
        """
        if not _TRANSFORMERS:
            raise RuntimeError("transformers not installed.")
        logger.info(f"Loading BERT NER model: {model_name} ...")
        self._pipe = hf_pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )
        logger.info("BERT NER model loaded.")

    def extract(self, text: str, sentence: str = "") -> list[Entity]:
        """Run BERT NER on text, return Entity list."""
        entities: list[Entity] = []
        # Chunk text to avoid exceeding model max length
        chunk_size = 450  # tokens ≈ chars (conservative)
        chunks = self._chunk_text(text, chunk_size)
        offset = 0
        for chunk in chunks:
            try:
                raw = self._pipe(chunk)
            except Exception as exc:
                logger.warning(f"BERT NER error on chunk: {exc}")
                offset += len(chunk)
                continue
            for item in raw:
                label = self.LABEL_MAP.get(item.get("entity_group", ""), "MISC")
                entities.append(Entity(
                    text=item["word"].strip(),
                    label=label,
                    label_desc=ENTITY_TYPES.get(label, label),
                    start=offset + item["start"],
                    end=offset + item["end"],
                    score=round(float(item.get("score", 1.0)), 4),
                    source="bert",
                    sentence=sentence,
                ))
            offset += len(chunk)
        return entities

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) < chunk_size:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks or [text]


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: Chemical / Process-Safety Rule-Based NER
# ══════════════════════════════════════════════════════════════════════════════

class ChemSafetyNERLayer:
    """
    Gazetteer + regex NER tuned for process-safety text.
    Zero-dependency, always available, runs on any text length.
    """

    def extract(self, text: str, sentence: str = "") -> list[Entity]:
        entities: list[Entity] = []
        for pattern, label in _RULE_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group().strip(),
                    label=label,
                    label_desc=ENTITY_TYPES.get(label, label),
                    start=match.start(),
                    end=match.end(),
                    score=0.85,
                    source="rule",
                    sentence=sentence,
                ))
        return entities


# ══════════════════════════════════════════════════════════════════════════════
# Merging & Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """
    Remove duplicate / overlapping entities.
    Priority: bert > rule (higher score wins on overlap).
    """
    # Sort by start position, then by descending score
    entities.sort(key=lambda e: (e.start, -e.score))
    kept: list[Entity] = []
    last_end = -1
    for ent in entities:
        if ent.start >= last_end:
            kept.append(ent)
            last_end = ent.end
        else:
            # Overlapping — keep highest score
            if ent.score > kept[-1].score:
                kept[-1] = ent
    return kept


def _normalise_text(text: str) -> str:
    """Lowercase + strip for dedup comparison."""
    return text.lower().strip()


def _remove_low_quality(entities: list[Entity], min_score: float = 0.5) -> list[Entity]:
    """Drop entities below confidence threshold and single-character hits."""
    return [e for e in entities if e.score >= min_score and len(e.text) > 1]


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class HAZOPNERPipeline:
    """
    Orchestrates BERT + rule-based NER over a full document.

    Usage:
        pipe = HAZOPNERPipeline(use_bert=True)
        results = pipe.run(document)   # document from ingestion.build_document()
        # results["entities"] → list[Entity]
        # results["by_label"] → dict[label → list[Entity]]
        # results["summary"]  → counts per label
    """

    def __init__(self, use_bert: bool = True, bert_device: int = -1):
        self.rule_layer = ChemSafetyNERLayer()
        self.bert_layer: Optional[BertNERLayer] = None
        if use_bert and _TRANSFORMERS:
            try:
                self.bert_layer = BertNERLayer(device=bert_device)
            except Exception as exc:
                logger.warning(f"Could not load BERT model: {exc}. Running rule-based NER only.")
        elif use_bert and not _TRANSFORMERS:
            logger.warning("transformers not installed — running rule-based NER only.")

    # ── Per-sentence extraction ───────────────────────────────────────────────

    def _extract_from_sentence(self, sentence: str, char_offset: int) -> list[Entity]:
        entities: list[Entity] = []

        # Rule-based layer
        rule_ents = self.rule_layer.extract(sentence, sentence=sentence)
        for ent in rule_ents:
            ent.start += char_offset
            ent.end   += char_offset
        entities.extend(rule_ents)

        # BERT layer
        if self.bert_layer:
            bert_ents = self.bert_layer.extract(sentence, sentence=sentence)
            for ent in bert_ents:
                ent.start += char_offset
                ent.end   += char_offset
            entities.extend(bert_ents)

        return entities

    # ── Full document run ─────────────────────────────────────────────────────

    def run(self, document: dict) -> dict:
        """
        Run NER over all sentences in a document dict (from ingestion module).
        Returns enriched result dict.
        """
        full_text = document.get("full_text", "")
        sentences = document.get("sentences", [full_text])

        all_entities: list[Entity] = []
        char_offset = 0

        for sent in sentences:
            ents = self._extract_from_sentence(sent, char_offset)
            all_entities.extend(ents)
            char_offset += len(sent) + 1  # +1 for separator

        # Post-process
        all_entities = _remove_low_quality(all_entities)
        all_entities = _deduplicate(all_entities)

        # Group by label
        by_label: dict[str, list[dict]] = {}
        for ent in all_entities:
            by_label.setdefault(ent.label, []).append(ent.to_dict())

        # Unique terms per label (for downstream use)
        unique_terms: dict[str, list[str]] = {
            label: list({_normalise_text(e["text"]) for e in ents})
            for label, ents in by_label.items()
        }

        summary = {label: len(ents) for label, ents in by_label.items()}
        total = sum(summary.values())

        logger.info(
            f"NER complete: {total} entities found "
            f"({', '.join(f'{k}={v}' for k, v in summary.items())})"
        )

        return {
            "metadata":     document.get("metadata", {}),
            "entities":     [e.to_dict() for e in all_entities],
            "by_label":     by_label,
            "unique_terms": unique_terms,
            "summary":      summary,
            "total":        total,
        }

    # ── Direct text shortcut ──────────────────────────────────────────────────

    def run_text(self, text: str) -> dict:
        """Convenience method for running NER on a plain string."""
        from src.ingestion import ingest_text
        doc = ingest_text(text, source_name="direct_input", save=False)
        return self.run(doc)


# ══════════════════════════════════════════════════════════════════════════════
# Entity Report Formatter (plain text)
# ══════════════════════════════════════════════════════════════════════════════

def format_ner_report(ner_result: dict) -> str:
    """
    Format NER results as a readable text report for quick review.
    """
    lines = ["=" * 60, "NER EXTRACTION REPORT", "=" * 60]
    meta = ner_result.get("metadata", {})
    lines.append(f"Source   : {meta.get('source', 'N/A')}")
    lines.append(f"Doc ID   : {meta.get('doc_id', 'N/A')}")
    lines.append(f"Date     : {meta.get('date', 'N/A')}")
    lines.append(f"Total entities: {ner_result.get('total', 0)}")
    lines.append("-" * 60)

    label_order = ["CHEM", "EQUIP", "PARAM", "CAUSE", "CONSEQ", "SAFEGUARD",
                   "LOC", "ORG", "PERSON", "MISC"]

    by_label = ner_result.get("by_label", {})
    unique   = ner_result.get("unique_terms", {})

    for label in label_order:
        if label not in by_label:
            continue
        desc  = ENTITY_TYPES.get(label, label)
        terms = unique.get(label, [])
        lines.append(f"\n[{label}] {desc} ({len(by_label[label])} hits)")
        lines.append("  Unique terms: " + ", ".join(sorted(terms)[:20]))

    lines.append("=" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Save / Load NER results
# ══════════════════════════════════════════════════════════════════════════════

def save_ner_result(ner_result: dict, output_dir: Optional[Path] = None) -> Path:
    from config import PROCESSED_DIR
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = ner_result.get("metadata", {}).get("doc_id", "unknown")
    out_path = output_dir / f"{doc_id}_ner.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(ner_result, fh, indent=2, ensure_ascii=False)
    logger.info(f"NER result saved → {out_path}")
    return out_path


def load_ner_result(doc_id: str, input_dir: Optional[Path] = None) -> dict:
    from config import PROCESSED_DIR
    input_dir = input_dir or PROCESSED_DIR
    path = input_dir / f"{doc_id}_ner.json"
    if not path.exists():
        raise FileNotFoundError(f"NER result not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from src.ingestion import ingest_file, ingest_text

    if len(sys.argv) < 2:
        # Quick smoke-test on a small inline text
        sample = (
            "A hydrogen sulfide leak from a corroded pipeline at the refinery "
            "caused an explosion resulting in two fatalities. "
            "The pressure relief valve had failed to open due to corrosion. "
            "Recommended safeguard: install gas detectors and upgrade to corrosion-resistant PRVs."
        )
        pipe = HAZOPNERPipeline(use_bert=False)  # rule-only for speed
        result = pipe.run_text(sample)
        print(format_ner_report(result))
    else:
        target = Path(sys.argv[1])
        doc = ingest_file(target, save=False)
        use_bert = "--no-bert" not in sys.argv
        pipe = HAZOPNERPipeline(use_bert=use_bert)
        result = pipe.run(doc)
        print(format_ner_report(result))
        save_ner_result(result)
