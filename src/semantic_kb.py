"""
semantic_kb.py — Embeddings-based Semantic Knowledge Base.

Replaces the hand-coded DEVIATION_KB dict with a vector search engine
over real incident reports and safety guidelines.

How it works:
  1. Index phase  — embed every sentence from processed incident documents
                    using sentence-transformers (all-MiniLM-L6-v2, 80 MB)
                    Store vectors in a FAISS index on disk
  2. Query phase  — embed a deviation description, find top-K similar
                    sentences, cluster by document, return structured results

Benefits over hand-coded KB:
  - Scales with data: add 500 CSB reports → 500× more knowledge
  - Semantic matching: "valve fails closed" matches "actuator seized shut"
  - No maintenance: new incident knowledge added by dropping a PDF in data/raw/
  - Confidence scores: cosine similarity tells you how relevant each hit is

Dependencies:
  sentence-transformers  (pip install sentence-transformers)
  faiss-cpu              (pip install faiss-cpu)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional
from loguru import logger

from config import PROCESSED_DIR, MODEL_DIR

INDEX_DIR     = MODEL_DIR / "semantic_kb"
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, 80 MB, excellent quality
EMBED_DIM     = 384
TOP_K_DEFAULT = 8


# ══════════════════════════════════════════════════════════════════════════════
# Embedding engine (lazy-loaded)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """Thin wrapper around sentence-transformers."""

    _instance = None

    @classmethod
    def get(cls) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            self._model = SentenceTransformer(EMBED_MODEL)
            logger.info("Embedding model loaded.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers faiss-cpu"
            )

    def embed(self, texts: list[str]) -> "np.ndarray":
        import numpy as np
        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return vecs.astype("float32")

    def embed_one(self, text: str) -> "np.ndarray":
        return self.embed([text])[0]


# ══════════════════════════════════════════════════════════════════════════════
# Document chunk store
# ══════════════════════════════════════════════════════════════════════════════

class ChunkStore:
    """
    Stores text chunks alongside their metadata.
    Each chunk is one sentence from a processed incident document.
    """

    def __init__(self):
        self.chunks:    list[str]  = []
        self.meta:      list[dict] = []   # {doc_id, source, date, label}

    def add(self, text: str, meta: dict):
        self.chunks.append(text)
        self.meta.append(meta)

    def get(self, idx: int) -> tuple[str, dict]:
        return self.chunks[idx], self.meta[idx]

    def __len__(self):
        return len(self.chunks)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"chunks": self.chunks, "meta": self.meta}, fh)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        store = cls()
        store.chunks = data["chunks"]
        store.meta   = data["meta"]
        return store


# ══════════════════════════════════════════════════════════════════════════════
# FAISS index wrapper
# ══════════════════════════════════════════════════════════════════════════════

class VectorIndex:
    """FAISS flat L2 index with cosine similarity (vectors pre-normalised)."""

    def __init__(self):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(EMBED_DIM)   # Inner product = cosine on normalised vecs
        except ImportError:
            raise RuntimeError("faiss-cpu not installed.\nRun: pip install faiss-cpu")
        self._faiss = faiss

    def add(self, vectors: "np.ndarray"):
        self._index.add(vectors)

    def search(self, query: "np.ndarray", k: int) -> tuple:
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        scores, indices = self._index.search(query, k)
        return scores[0], indices[0]

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed.")
        vi = cls.__new__(cls)
        vi._faiss  = faiss
        vi._index  = faiss.read_index(str(path))
        return vi


# ══════════════════════════════════════════════════════════════════════════════
# Semantic KB — main class
# ══════════════════════════════════════════════════════════════════════════════

class SemanticKB:
    """
    Vector-search knowledge base over incident reports.

    Usage:
        kb = SemanticKB()
        kb.build()                          # index all processed documents
        results = kb.search("cooling water valve fails closed", top_k=8)
        # results → list of SemanticHit
    """

    INDEX_FILE = INDEX_DIR / "faiss.index"
    STORE_FILE = INDEX_DIR / "chunks.pkl"
    META_FILE  = INDEX_DIR / "kb_meta.json"

    def __init__(self):
        self._engine: Optional[EmbeddingEngine] = None
        self._index:  Optional[VectorIndex]     = None
        self._store:  Optional[ChunkStore]      = None
        self._built   = False

    # ── Build / index ─────────────────────────────────────────────────────────

    def build(
        self,
        processed_dir: Optional[Path] = None,
        include_kb_text: bool = True,
        force_rebuild:   bool = False,
    ) -> int:
        """
        Index all processed documents. Returns number of chunks indexed.
        Skips rebuild if index already exists (unless force_rebuild=True).
        """
        if self.INDEX_FILE.exists() and not force_rebuild:
            logger.info("Semantic KB index found — loading from disk.")
            self._load()
            return self._store.__len__() if self._store else 0

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        processed_dir = processed_dir or PROCESSED_DIR

        self._engine = EmbeddingEngine.get()
        self._store  = ChunkStore()
        self._index  = VectorIndex()

        # Collect chunks from processed documents
        doc_files = list(processed_dir.glob("*.json"))
        doc_files  = [f for f in doc_files if not f.stem.endswith("_ner")]

        logger.info(f"Indexing {len(doc_files)} documents…")
        all_texts: list[str] = []
        all_meta:  list[dict] = []

        for doc_path in doc_files:
            try:
                with open(doc_path, encoding="utf-8") as fh:
                    doc = json.load(fh)
                meta_base = {
                    "doc_id":  doc.get("metadata", {}).get("doc_id", ""),
                    "source":  doc.get("metadata", {}).get("source", ""),
                    "date":    doc.get("metadata", {}).get("date", ""),
                }
                for sent in doc.get("sentences", []):
                    if len(sent) > 30:
                        all_texts.append(sent)
                        all_meta.append({**meta_base, "type": "sentence"})
            except Exception as exc:
                logger.warning(f"Failed to index {doc_path.name}: {exc}")

        # Also index the hand-coded KB as seed knowledge
        if include_kb_text:
            kb_texts, kb_meta = self._kb_chunks()
            all_texts.extend(kb_texts)
            all_meta.extend(kb_meta)

        if not all_texts:
            logger.warning("No text to index. Run ingestion first or add reports to data/raw/")
            return 0

        logger.info(f"Embedding {len(all_texts)} chunks…")
        vectors = self._engine.embed(all_texts)

        for text, meta in zip(all_texts, all_meta):
            self._store.add(text, meta)
        self._index.add(vectors)

        # Persist
        self._index.save(self.INDEX_FILE)
        self._store.save(self.STORE_FILE)
        with open(self.META_FILE, "w") as fh:
            json.dump({"total": len(all_texts), "docs": len(doc_files)}, fh)

        self._built = True
        logger.info(f"Semantic KB built: {len(all_texts)} chunks indexed.")
        return len(all_texts)

    def _load(self):
        self._engine = EmbeddingEngine.get()
        self._index  = VectorIndex.load(self.INDEX_FILE)
        self._store  = ChunkStore.load(self.STORE_FILE)
        self._built  = True
        logger.info(f"Semantic KB loaded: {self._store.__len__()} chunks")

    def _kb_chunks(self) -> tuple[list[str], list[dict]]:
        """Convert hand-coded DEVIATION_KB into indexable text."""
        try:
            from src.hazop_engine import DEVIATION_KB
        except Exception:
            return [], []
        texts, metas = [], []
        for (gw, param), entry in DEVIATION_KB.items():
            for field, items in [
                ("cause", entry.get("causes", [])),
                ("consequence", entry.get("consequences", [])),
                ("safeguard", entry.get("safeguards", []) + entry.get("recommended", [])),
            ]:
                for item in items:
                    texts.append(f"{gw} {param}: {item}")
                    metas.append({
                        "doc_id": "kb", "source": "HAZOP KB",
                        "date": "built-in", "type": field,
                        "guide_word": gw, "parameter": param,
                        "ref": entry.get("ref", ""),
                    })
        return texts, metas

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        min_score: float = 0.3,
        filter_type: Optional[str] = None,   # "cause"|"consequence"|"safeguard"|None
    ) -> list[dict]:
        """
        Semantic search over the KB.
        Returns list of hit dicts sorted by relevance score.
        """
        if not self._built:
            self.build()

        if self._index.ntotal == 0:
            logger.warning("Empty index — returning empty results")
            return []

        query_vec = self._engine.embed_one(query)
        scores, indices = self._index.search(query_vec, min(top_k * 3, self._index.ntotal))

        hits = []
        for score, idx in zip(scores, indices):
            if idx < 0 or float(score) < min_score:
                continue
            text, meta = self._store.get(int(idx))
            if filter_type and meta.get("type") != filter_type:
                continue
            hits.append({
                "text":      text,
                "score":     round(float(score), 4),
                "doc_id":    meta.get("doc_id", ""),
                "source":    meta.get("source", ""),
                "date":      meta.get("date", ""),
                "type":      meta.get("type", ""),
                "ref":       meta.get("ref", ""),
                "guide_word":meta.get("guide_word", ""),
                "parameter": meta.get("parameter", ""),
            })
            if len(hits) >= top_k:
                break

        return hits

    def search_deviation(self, guide_word: str, parameter: str,
                          chemical: str = "") -> dict:
        """
        High-level search that returns structured causes/consequences/safeguards
        for a given HAZOP deviation. Drop-in enhancement for hazop_engine KB lookup.
        """
        query = f"{guide_word} {parameter} {chemical}".strip()

        causes     = self.search(query + " cause failure initiating event",    filter_type="cause",       top_k=5)
        conseqs    = self.search(query + " consequence outcome explosion fire", filter_type="consequence", top_k=4)
        safeguards = self.search(query + " safeguard protection barrier",      filter_type="safeguard",   top_k=4)

        return {
            "causes":      [h["text"] for h in causes],
            "consequences":[h["text"] for h in conseqs],
            "safeguards":  [h["text"] for h in safeguards],
            "sources":     list({h["source"] for h in causes + conseqs + safeguards}),
            "top_score":   max((h["score"] for h in causes + conseqs + safeguards), default=0.0),
        }

    def find_similar_incidents(self, scenario_description: str, top_k: int = 5) -> list[dict]:
        """
        Given a scenario description, find the most similar historical incidents.
        Used in the Scenario Browser for historical precedent lookup.
        """
        hits = self.search(scenario_description, top_k=top_k * 2)
        # Group by document
        by_doc: dict[str, dict] = {}
        for h in hits:
            doc_id = h["doc_id"]
            if doc_id == "kb":
                continue
            if doc_id not in by_doc or h["score"] > by_doc[doc_id]["score"]:
                by_doc[doc_id] = h
        return sorted(by_doc.values(), key=lambda x: -x["score"])[:top_k]

    @property
    def is_ready(self) -> bool:
        return self._built and self._index is not None and self._index.ntotal > 0

    def stats(self) -> dict:
        if not self._built:
            return {"status": "not built"}
        return {
            "status":  "ready",
            "chunks":  self._store.__len__() if self._store else 0,
            "vectors": self._index.ntotal if self._index else 0,
            "model":   EMBED_MODEL,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Singleton accessor
# ══════════════════════════════════════════════════════════════════════════════

_kb_instance: Optional[SemanticKB] = None

def get_semantic_kb(auto_build: bool = True) -> SemanticKB:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = SemanticKB()
        if auto_build:
            try:
                _kb_instance.build()
            except Exception as exc:
                logger.warning(f"Semantic KB build failed: {exc}. "
                               "Install sentence-transformers and faiss-cpu.")
    return _kb_instance


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    kb = SemanticKB()

    if "--build" in sys.argv or "--rebuild" in sys.argv:
        n = kb.build(force_rebuild="--rebuild" in sys.argv)
        print(f"Indexed {n} chunks")
    else:
        kb.build()
        print(f"KB stats: {kb.stats()}")

    query = " ".join(sys.argv[1:]) or "cooling water valve fails closed reactor temperature rises"
    print(f"\nQuery: {query}")
    hits = kb.search(query, top_k=5)
    for i, h in enumerate(hits, 1):
        print(f"\n  {i}. [{h['score']:.3f}] {h['source']}")
        print(f"     {h['text'][:120]}")
