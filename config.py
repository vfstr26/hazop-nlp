"""
config.py — Central configuration for HAZOP NLP system.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
SAMPLE_DIR       = DATA_DIR / "sample_reports"
MODEL_DIR        = BASE_DIR / "models"
OUTPUT_DIR       = BASE_DIR / "outputs"

# ── NER / BERT ─────────────────────────────────────────────────────────────────
NER_MODEL_NAME   = "dslim/bert-base-NER"          # HuggingFace NER model
SPACY_MODEL      = "en_core_web_sm"               # fallback spaCy model
NER_BATCH_SIZE   = 16
NER_MAX_LENGTH   = 512

# ── LLM (OpenAI-compatible endpoint or local) ──────────────────────────────────
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "local"
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(MODEL_DIR / "llm" / "model.gguf"))
LLM_TEMPERATURE  = 0.1
LLM_MAX_TOKENS   = 1500

# ── HAZOP Guide Words ──────────────────────────────────────────────────────────
HAZOP_GUIDE_WORDS = [
    "No/None", "More", "Less", "As Well As",
    "Part Of", "Reverse", "Other Than", "Early",
    "Late", "Before", "After"
]

# ── Chemical Entity Types (BERT NER extension labels) ─────────────────────────
ENTITY_TYPES = {
    "CHEM":      "Chemical / Substance",
    "EQUIP":     "Equipment / Component",
    "CAUSE":     "Cause / Initiating Event",
    "CONSEQ":    "Consequence / Outcome",
    "SAFEGUARD": "Safeguard / Mitigation",
    "PARAM":     "Process Parameter",
    "LOC":       "Location / Plant Area",
    "ORG":       "Organisation",
    "PERSON":    "Person / Role",
}

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_FORMATS = ["json", "html", "excel", "csv"]
DEFAULT_OUTPUT  = "html"
