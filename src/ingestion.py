"""
ingestion.py — Data ingestion module for HAZOP NLP system.

Handles:
  - PDF parsing (CSB reports, EPSC documents)
  - Plain text / .txt ingestion
  - Word (.docx) ingestion
  - URL/web scraping (CSB website incident listing)
  - Batch processing of a directory
  - Text cleaning and sentence segmentation
  - Saving processed output to data/processed/

Supported sources:
  • U.S. Chemical Safety Board  — https://www.csb.gov/investigations/
  • AIChE EPSC / CCPS databases  — local PDF / text files
"""

import os
import re
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from loguru import logger
from tqdm import tqdm

# ── Optional heavy imports (graceful degradation) ─────────────────────────────
try:
    import pdfplumber
    _PDFPLUMBER = True
except ImportError:
    _PDFPLUMBER = False
    logger.warning("pdfplumber not installed — PDF support disabled.")

try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False

try:
    from docx import Document as DocxDocument
    _DOCX = True
except ImportError:
    _DOCX = False
    logger.warning("python-docx not installed — .docx support disabled.")

from config import RAW_DIR, PROCESSED_DIR, SAMPLE_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
CSB_BASE_URL = "https://www.csb.gov"
CSB_INCIDENTS_URL = "https://www.csb.gov/investigations/"
REQUEST_HEADERS = {
    "User-Agent": "HAZOP-NLP-Research-Tool/1.0 (academic; safety research)"
}
REQUEST_TIMEOUT = 30  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Text Cleaning
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Normalise raw extracted text.
      - Remove control characters and ligatures
      - Collapse excessive whitespace / blank lines
      - Fix hyphenated line-breaks from PDF columns
      - Preserve paragraph breaks
    """
    if not text:
        return ""

    # Replace common PDF ligatures
    replacements = {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u00a0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Fix hyphenated line-breaks: "reac-\ntion" → "reaction"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Preserve paragraph boundaries (two+ newlines → double newline)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def segment_sentences(text: str) -> list[str]:
    """
    Naive sentence segmentation good enough for safety reports.
    Falls back to period-based splitting when spaCy is unavailable.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")
        doc = nlp(text[:100_000])  # cap for very large docs
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        # Fallback: split on ". " followed by capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# File Parsers
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(filepath: Union[str, Path]) -> str:
    """
    Extract text from a PDF using pdfplumber (preferred) or PyMuPDF.
    Returns concatenated page text.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    text_pages: list[str] = []

    if _PDFPLUMBER:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        logger.debug(f"pdfplumber extracted {len(text_pages)} pages from {filepath.name}")
    elif _PYMUPDF:
        doc = fitz.open(str(filepath))
        for page in doc:
            text_pages.append(page.get_text())
        doc.close()
        logger.debug(f"PyMuPDF extracted {len(text_pages)} pages from {filepath.name}")
    else:
        raise RuntimeError("No PDF library available. Install pdfplumber or PyMuPDF.")

    return clean_text("\n\n".join(text_pages))


def parse_txt(filepath: Union[str, Path]) -> str:
    """Read and clean a plain-text file."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    return clean_text(text)


def parse_docx(filepath: Union[str, Path]) -> str:
    """Extract text from a .docx Word document."""
    if not _DOCX:
        raise RuntimeError("python-docx not installed.")
    filepath = Path(filepath)
    doc = DocxDocument(str(filepath))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n\n".join(paragraphs))


def parse_file(filepath: Union[str, Path]) -> str:
    """
    Auto-detect file type and dispatch to the correct parser.
    Supports: .pdf, .txt, .docx
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    dispatch = {
        ".pdf":  parse_pdf,
        ".txt":  parse_txt,
        ".docx": parse_docx,
    }
    if suffix not in dispatch:
        raise ValueError(f"Unsupported file type: {suffix}")
    return dispatch[suffix](filepath)


# ══════════════════════════════════════════════════════════════════════════════
# Metadata Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_metadata(text: str, source_name: str) -> dict:
    """
    Heuristically pull report metadata from the first ~500 chars of the text.
    Returns a dict with: title, date, location, chemicals, doc_id.
    """
    snippet = text[:500]

    # Date: match patterns like "March 23, 2005" or "2005-03-23"
    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|' \
                   r'September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
    dates = re.findall(date_pattern, snippet, flags=re.IGNORECASE)

    # Location: "in <City, State>" or "at <Facility>"
    loc_pattern = r'(?:in|at|near)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z]{2})?)'
    locations = re.findall(loc_pattern, snippet)

    # Common chemical names in the snippet
    chem_pattern = r'\b(?:hydrogen\s+sulfide|ammonia|chlorine|propane|ethylene|' \
                   r'methane|gasoline|sulfuric acid|hydrochloric acid|benzene|' \
                   r'natural gas|butane|ethanol|acetylene)\b'
    chemicals = list(set(re.findall(chem_pattern, text[:2000], flags=re.IGNORECASE)))

    return {
        "source":     source_name,
        "doc_id":     hashlib.md5(text[:200].encode()).hexdigest()[:10],
        "date":       dates[0] if dates else "Unknown",
        "location":   locations[0].strip() if locations else "Unknown",
        "chemicals":  chemicals[:5],
        "char_count": len(text),
        "ingested_at": datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Processed Document Structure
# ══════════════════════════════════════════════════════════════════════════════

def build_document(text: str, metadata: dict) -> dict:
    """
    Wrap cleaned text + metadata into a standard document dict
    that all downstream modules consume.
    """
    return {
        "metadata":  metadata,
        "full_text": text,
        "sentences": segment_sentences(text),
        "paragraphs": [p.strip() for p in text.split("\n\n") if p.strip()],
    }


def save_processed(document: dict, output_dir: Optional[Path] = None) -> Path:
    """Persist a processed document as JSON to data/processed/."""
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_id = document["metadata"]["doc_id"]
    out_path = output_dir / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved processed document → {out_path}")
    return out_path


def load_processed(doc_id: str, input_dir: Optional[Path] = None) -> dict:
    """Load a previously saved processed document by doc_id."""
    input_dir = input_dir or PROCESSED_DIR
    path = input_dir / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Processed document not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# Batch Processing
# ══════════════════════════════════════════════════════════════════════════════

def ingest_file(filepath: Union[str, Path], save: bool = True) -> dict:
    """
    Full pipeline for a single file:
      parse → clean → segment → metadata → save (optional)
    """
    filepath = Path(filepath)
    logger.info(f"Ingesting: {filepath.name}")
    text = parse_file(filepath)
    meta = extract_metadata(text, source_name=filepath.name)
    doc  = build_document(text, meta)
    if save:
        save_processed(doc)
    return doc


def ingest_directory(
    directory: Union[str, Path] = None,
    extensions: tuple = (".pdf", ".txt", ".docx"),
    save: bool = True,
) -> list[dict]:
    """
    Ingest all supported files in a directory.
    Returns list of processed document dicts.
    """
    directory = Path(directory) if directory else RAW_DIR
    files = [f for f in directory.iterdir() if f.suffix.lower() in extensions]
    if not files:
        logger.warning(f"No supported files found in {directory}")
        return []

    results = []
    for f in tqdm(files, desc="Ingesting files"):
        try:
            doc = ingest_file(f, save=save)
            results.append(doc)
        except Exception as exc:
            logger.error(f"Failed to ingest {f.name}: {exc}")

    logger.info(f"Ingestion complete: {len(results)}/{len(files)} files processed.")
    return results


def ingest_text(raw_text: str, source_name: str = "manual_input", save: bool = False) -> dict:
    """
    Ingest raw text pasted directly (e.g., from the Streamlit UI).
    """
    text = clean_text(raw_text)
    meta = extract_metadata(text, source_name=source_name)
    return build_document(text, meta)


# ══════════════════════════════════════════════════════════════════════════════
# CSB Web Scraper (incident index)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_csb_incident_list(max_pages: int = 3) -> list[dict]:
    """
    Scrape the CSB investigations page for incident titles, dates, and URLs.
    Returns a list of dicts with keys: title, date, url, chemicals.

    NOTE: This is read-only, respectful of rate limits, and for academic use.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 not installed. Run: pip install beautifulsoup4")
        return []

    incidents: list[dict] = []
    page = 1

    while page <= max_pages:
        url = f"{CSB_INCIDENTS_URL}?page={page}"
        logger.info(f"Fetching CSB page {page}: {url}")
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Request failed: {exc}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select(".investigation-list-item, .investigation-card, article")

        if not cards:
            logger.info("No more incident cards found.")
            break

        for card in cards:
            title_el = card.find(["h2", "h3", "h4"])
            link_el  = card.find("a", href=True)
            date_el  = card.find(class_=re.compile(r"date|time", re.I))

            title = title_el.get_text(strip=True) if title_el else "Unknown"
            href  = link_el["href"] if link_el else ""
            url_  = f"{CSB_BASE_URL}{href}" if href.startswith("/") else href
            date  = date_el.get_text(strip=True) if date_el else "Unknown"

            incidents.append({"title": title, "date": date, "url": url_})

        page += 1

    logger.info(f"Found {len(incidents)} CSB incidents.")
    return incidents


def download_csb_report(url: str, dest_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download a single CSB PDF report from the given URL.
    Returns the saved file path or None on failure.
    """
    dest_dir = dest_dir or RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading: {url}")
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        # Derive filename
        filename = url.rstrip("/").split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        dest_path = dest_dir / filename
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

        logger.info(f"Saved to {dest_path}")
        return dest_path

    except requests.RequestException as exc:
        logger.error(f"Download failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Quick test / CLI entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        docs = ingest_directory(target)
        print(f"Ingested {len(docs)} documents.")
    elif target.is_file():
        doc = ingest_file(target)
        print(f"Title  : {doc['metadata']['source']}")
        print(f"Doc ID : {doc['metadata']['doc_id']}")
        print(f"Date   : {doc['metadata']['date']}")
        print(f"Chars  : {doc['metadata']['char_count']}")
        print(f"Sentences: {len(doc['sentences'])}")
    else:
        print(f"Path not found: {target}")
        sys.exit(1)
