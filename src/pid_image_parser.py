"""
pid_image_parser.py — Vision LLM P&ID image parser.

Accepts scanned P&ID drawings (PNG, JPG, PDF) and extracts:
  - Equipment items (tag, name, type)
  - Stream connections
  - Instruments and safeguards
  - Operating conditions from annotation boxes

Uses GPT-4o vision (or any OpenAI-compatible vision endpoint).
Falls back to text extraction (pdfplumber OCR) for PDFs.

Workflow:
  1. Convert image/PDF to base64
  2. Send to vision LLM with a structured extraction prompt
  3. Parse JSON response into PIDSystem
  4. Optional: overlay extraction results on the original image
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Optional, Union
from loguru import logger

from src.pid_parser import PIDSystem, parse_json
from config import OPENAI_API_KEY, OPENAI_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# Vision extraction prompt
# ══════════════════════════════════════════════════════════════════════════════

PID_VISION_PROMPT = """
You are an expert process engineer analysing a Piping and Instrumentation Diagram (P&ID).

Carefully examine this P&ID drawing and extract ALL of the following:

1. Every equipment item: tag number (e.g. V-101, R-201, E-301), name, and type
2. Every instrument: tag (e.g. FIC-101, TAHH-201, PSV-301), type, and function
3. Every process stream connection: which equipment connects to which
4. Any visible operating conditions: temperatures, pressures, flow rates
5. Any visible safeguards: relief valves, interlocks, ESD systems

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{
  "system_name": "Name or description of the process system",
  "utilities": ["cooling water", "steam", ...],
  "nodes": [
    {
      "tag": "V-101",
      "name": "Feed Drum",
      "node_type": "Vessel / Tank",
      "chemicals": ["propane"],
      "temperature_c": 40,
      "pressure_barg": 5.0,
      "design_pressure_barg": 10.0,
      "phase": "liquid",
      "moc": "carbon steel",
      "safeguards": ["PSV-101", "LAH-101"],
      "description": "Feed surge drum"
    }
  ],
  "streams": [
    {
      "stream_id": "S-01",
      "from_tag": "V-101",
      "to_tag": "P-101",
      "chemicals": ["propane"],
      "phase": "liquid",
      "has_check_valve": false
    }
  ]
}

Node type must be one of:
Reactor, Vessel / Tank, Distillation Column, Heat Exchanger, Pump,
Compressor, Valve, Pipe / Line, Instrument, Separator, Furnace / Fired Heater,
Storage Tank, Scrubber / Absorber, Filter / Strainer, Utility System, Other

If you cannot read a tag or value clearly, use "?" for the tag and omit numeric values.
Include every piece of equipment visible, even if details are unclear.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Image utilities
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
SUPPORTED_DOC_TYPES   = {".pdf"}
MAX_IMAGE_SIZE_MB     = 20


def _file_to_base64(path: Path) -> tuple[str, str]:
    """
    Convert an image file to base64 string.
    Returns (base64_string, mime_type).
    """
    suffix = path.suffix.lower()
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/png",   # convert BMP via PIL if needed
    }
    mime_type = mime_map.get(suffix, "image/png")

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(f"Image too large: {size_mb:.1f} MB (max {MAX_IMAGE_SIZE_MB} MB). "
                         "Please resize to under 20 MB.")

    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")

    return b64, mime_type


def _pdf_page_to_base64(path: Path, page: int = 0) -> tuple[str, str]:
    """
    Convert a PDF page to a PNG image, then base64-encode it.
    Requires PyMuPDF (fitz).
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF not installed.\nRun: pip install PyMuPDF")

    doc  = fitz.open(str(path))
    pg   = doc[min(page, len(doc) - 1)]
    mat  = fitz.Matrix(2.0, 2.0)   # 2× zoom for better resolution
    pix  = pg.get_pixmap(matrix=mat)
    png  = pix.tobytes("png")
    doc.close()

    b64 = base64.b64encode(png).decode("utf-8")
    return b64, "image/png"


def _bytes_to_base64(image_bytes: bytes, mime_type: str = "image/png") -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Vision LLM extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_with_vision_llm(
    b64_image:  str,
    mime_type:  str,
    api_key:    str,
    model:      str = "gpt-4o",
    base_url:   Optional[str] = None,
) -> dict:
    """
    Send image to OpenAI vision API and parse the JSON response.
    Returns raw parsed dict (not yet a PIDSystem).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai not installed.\nRun: pip install openai")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    logger.info(f"Sending P&ID image to vision LLM ({model})…")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": PID_VISION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url":    f"data:{mime_type};base64,{b64_image}",
                        "detail": "high",
                    }},
                ],
            }
        ],
        max_tokens=4000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content or ""
    logger.info(f"Vision LLM response: {len(raw)} chars")

    # Strip markdown fences
    raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
    raw = re.sub(r'\n?```$', '', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc}. Attempting recovery…")
        # Try extracting JSON object from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse vision LLM response as JSON: {raw[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF text fallback (for when vision LLM is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_from_pdf_text(path: Path) -> PIDSystem:
    """
    Fallback: extract text from PDF and parse with rule-based parser.
    Not as good as vision but works without an API key.
    """
    from src.ingestion   import parse_pdf
    from src.pid_parser  import parse_text
    logger.info("Using text-based PDF extraction (vision LLM unavailable)")
    text = parse_pdf(path)
    return parse_text(text, system_name=path.stem)


# ══════════════════════════════════════════════════════════════════════════════
# Main parser
# ══════════════════════════════════════════════════════════════════════════════

class PIDImageParser:
    """
    Parse a P&ID image or PDF into a PIDSystem using vision LLM.

    Usage:
        parser = PIDImageParser()
        pid = parser.parse("plant_pid.png")
        pid = parser.parse("drawing.pdf", page=0)
        pid = parser.parse(image_bytes, mime_type="image/png")
    """

    def __init__(
        self,
        api_key:  str = OPENAI_API_KEY,
        model:    str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        self.api_key  = api_key
        self.model    = model
        self.base_url = base_url
        self._available = bool(api_key)

        if not self._available:
            logger.warning(
                "No OpenAI API key — vision P&ID parsing unavailable. "
                "Set OPENAI_API_KEY in .env. Will fall back to text extraction for PDFs."
            )

    def parse(
        self,
        source:      Union[str, Path, bytes],
        mime_type:   str = "image/png",
        page:        int = 0,
        system_name: str = "P&ID System",
    ) -> PIDSystem:
        """
        Parse a P&ID image/PDF into a PIDSystem.

        Parameters
        ----------
        source    : file path (str/Path), bytes, or base64 string
        mime_type : MIME type if source is bytes
        page      : PDF page number (0-indexed)
        """
        # ── Resolve source to (b64, mime_type) ───────────────────────────────
        if isinstance(source, bytes):
            b64   = _bytes_to_base64(source, mime_type)
            mtype = mime_type

        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            suffix = path.suffix.lower()

            if suffix in SUPPORTED_DOC_TYPES:
                if not self._available:
                    return _extract_from_pdf_text(path)
                b64, mtype = _pdf_page_to_base64(path, page)

            elif suffix in SUPPORTED_IMAGE_TYPES:
                if not self._available:
                    raise RuntimeError(
                        "OpenAI API key required for image parsing. "
                        "Set OPENAI_API_KEY in .env"
                    )
                b64, mtype = _file_to_base64(path)

            else:
                raise ValueError(f"Unsupported file type: {suffix}")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # ── Vision extraction ─────────────────────────────────────────────────
        raw_data = _extract_with_vision_llm(
            b64, mtype, self.api_key, self.model, self.base_url
        )

        raw_data.setdefault("system_name", system_name)

        # ── Build PIDSystem ───────────────────────────────────────────────────
        pid = parse_json(raw_data)
        logger.info(
            f"Vision parse complete: {len(pid.nodes)} nodes, "
            f"{len(pid.streams)} streams extracted from image"
        )
        return pid

    def parse_bytes(self, image_bytes: bytes,
                    mime_type: str = "image/png") -> PIDSystem:
        """Parse image bytes directly (used from Streamlit file uploader)."""
        return self.parse(image_bytes, mime_type=mime_type)

    @property
    def is_available(self) -> bool:
        return self._available


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit upload widget helper
# ══════════════════════════════════════════════════════════════════════════════

def render_pid_image_uploader(parser: Optional[PIDImageParser] = None):
    """
    Drop-in Streamlit widget for P&ID image upload and parsing.
    Returns PIDSystem or None.
    """
    import streamlit as st

    parser = parser or PIDImageParser()

    if not parser.is_available:
        st.warning(
            "⚠ Vision P&ID parsing requires an OpenAI API key (GPT-4o).  \n"
            "Add `OPENAI_API_KEY` to your `.env` file to enable this feature.  \n"
            "You can still use the **text description** input without an API key."
        )
        return None

    st.markdown(
        "Upload a scanned P&ID drawing. Supported formats: "
        "**PNG, JPG, PDF** (max 20 MB). "
        "GPT-4o vision will extract all equipment, instruments, and connections."
    )

    uploaded = st.file_uploader(
        "P&ID drawing",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed",
    )

    pdf_page = 0
    if uploaded and uploaded.name.lower().endswith(".pdf"):
        pdf_page = st.number_input(
            "PDF page number (0 = first page)", min_value=0, value=0
        )

    if uploaded and st.button("🔍 Extract P&ID from Image", type="primary"):
        with st.spinner("Sending to GPT-4o vision… (this takes 10–30 seconds)"):
            try:
                image_bytes = uploaded.read()
                suffix      = Path(uploaded.name).suffix.lower()

                if suffix == ".pdf":
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = Path(tmp.name)
                    pid = parser.parse(tmp_path, page=pdf_page,
                                       system_name=uploaded.name)
                    tmp_path.unlink(missing_ok=True)
                else:
                    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg"}
                    mime = mime_map.get(suffix, "image/png")
                    pid  = parser.parse_bytes(image_bytes, mime_type=mime)

                st.success(
                    f"✅ Extracted **{len(pid.nodes)} nodes** and "
                    f"**{len(pid.streams)} streams** from the P&ID drawing."
                )
                return pid

            except Exception as exc:
                st.error(f"Vision extraction failed: {exc}")
                return None

    return None
