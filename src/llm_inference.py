"""
llm_inference.py — LLM inference module for HAZOP NLP system.

Provides two modes:
  1. OpenAI-compatible API  (GPT-4o-mini, GPT-4, any OpenAI-API endpoint)
  2. Local GGUF model       (llama-cpp-python — Mistral, Llama-3, etc.)

Primary use-cases:
  A. Enrich HAZOP rows — use LLM to refine causes, consequences, safeguards
     based on the actual incident text context.
  B. Generate narrative summary of HAZOP findings.
  C. Answer Q&A over the incident text ("What caused the explosion?").
  D. Rate / validate NER extractions.

All calls are wrapped with retry logic, token-counting, and cost estimation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Generator

from loguru import logger
from config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    LOCAL_MODEL_PATH, LLM_TEMPERATURE, LLM_MAX_TOKENS
)


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert process safety engineer with 20+ years of experience 
conducting HAZOP studies, incident investigations, and process risk assessments. 
You are familiar with IEC 61882, CCPS Guidelines, CSB accident reports, and AIChE EPSC best practices.

When asked to analyse a process description or incident report, you:
- Apply HAZOP guide words systematically (No/None, More, Less, Reverse, As Well As, Other Than)
- Reference real historical incidents where relevant
- Provide specific, actionable safeguard recommendations
- Use concise, structured language suitable for safety worksheets
- Flag critical risks clearly with URGENT labels
- NEVER hallucinate chemical data — if unsure, say so
"""

HAZOP_ENRICHMENT_TEMPLATE = """
Given the following HAZOP deviation and process context, provide an enhanced analysis.

**Process Description / Incident Context:**
{context}

**Existing HAZOP Row:**
- Node: {node_id}
- Equipment: {equipment}
- Chemical: {chemical}  
- Deviation: {deviation}
- Current Causes: {causes}
- Current Consequences: {consequences}
- Current Safeguards: {safeguards}

Please provide a JSON response with this exact structure:
{{
  "enriched_causes": ["cause 1", "cause 2", "cause 3"],
  "enriched_consequences": ["consequence 1", "consequence 2"],
  "enriched_safeguards_existing": ["safeguard 1", "safeguard 2"],
  "enriched_safeguards_recommended": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "severity": <1-5>,
  "likelihood": <1-5>,
  "historical_incidents": ["CSB/EPSC reference if applicable"],
  "narrative_notes": "Brief expert commentary on this deviation in context."
}}
Only output valid JSON. No markdown fences.
"""

SUMMARY_TEMPLATE = """
Based on the following HAZOP analysis results, write a professional executive summary 
for a process safety engineer audience.

**Analysis Statistics:**
{stats}

**Top Critical/High Risk Deviations:**
{top_rows}

**Incident Context:**
{context}

Write a 3-4 paragraph executive summary covering:
1. Overview of the process and what was analysed
2. Key hazards identified and their risk levels
3. Most critical safeguard gaps
4. Recommended immediate actions

Keep the tone professional, factual, and concise. Use bullet points where appropriate.
"""

QA_TEMPLATE = """
Using only the information in the following incident report excerpt, answer the question.
If the answer is not in the text, say "Not found in the provided text."

**Incident Report:**
{context}

**Question:** {question}

Provide a precise, factual answer with relevant quotes from the text where possible.
"""

NER_VALIDATION_TEMPLATE = """
Review the following Named Entity Recognition extractions from an incident report.
Identify any errors, missed entities, or misclassifications.

**Original text:**
{text}

**Extracted entities:**
{entities}

Respond with JSON:
{{
  "validated": [
    {{"text": "...", "label": "CHEM|EQUIP|CAUSE|CONSEQ|SAFEGUARD|PARAM", "correct": true/false, "correction": "if wrong, correct label or text"}}
  ],
  "missed_entities": [
    {{"text": "...", "label": "...", "reason": "why this was missed"}}
  ],
  "overall_quality": "Good|Fair|Poor",
  "notes": "..."
}}
Only output valid JSON.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Token & Cost Estimation
# ══════════════════════════════════════════════════════════════════════════════

# Approximate token cost per 1K tokens (USD) — update as pricing changes
MODEL_COSTS = {
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.00060},
    "gpt-4o":       {"input": 0.005,   "output": 0.015},
    "gpt-4":        {"input": 0.03,    "output": 0.06},
    "gpt-3.5-turbo":{"input": 0.0005,  "output": 0.0015},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def estimate_cost(input_text: str, output_text: str, model: str = OPENAI_MODEL) -> float:
    costs = MODEL_COSTS.get(model, {"input": 0.002, "output": 0.002})
    in_k  = estimate_tokens(input_text)  / 1000
    out_k = estimate_tokens(output_text) / 1000
    return round(in_k * costs["input"] + out_k * costs["output"], 6)


# ══════════════════════════════════════════════════════════════════════════════
# Base LLM Client
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Abstract base — subclassed by OpenAIClient and LocalClient."""

    def complete(self, prompt: str, system: str = SYSTEM_PROMPT,
                 temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = LLM_MAX_TOKENS) -> str:
        raise NotImplementedError

    def complete_json(self, prompt: str, system: str = SYSTEM_PROMPT,
                      retries: int = 3) -> dict:
        """Call complete() and parse JSON response, with retries on parse failure."""
        for attempt in range(1, retries + 1):
            raw = self.complete(prompt, system=system)
            try:
                # Strip markdown fences if present
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = re.sub(r"^```[a-z]*\n?", "", clean)
                    clean = re.sub(r"\n?```$", "", clean)
                return json.loads(clean)
            except json.JSONDecodeError as exc:
                logger.warning(f"JSON parse failed (attempt {attempt}/{retries}): {exc}")
                if attempt == retries:
                    logger.error("All JSON parse attempts failed. Returning raw text.")
                    return {"raw_response": raw, "parse_error": str(exc)}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI Client
# ══════════════════════════════════════════════════════════════════════════════

class OpenAIClient(LLMClient):
    """
    OpenAI API client (also works with Azure OpenAI, Groq, Ollama with OpenAI-compat endpoint).
    """

    def __init__(
        self,
        api_key:   str = OPENAI_API_KEY,
        model:     str = OPENAI_MODEL,
        base_url:  Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        if not api_key:
            raise ValueError(
                "No OpenAI API key. Set OPENAI_API_KEY in .env or environment."
            )

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model  = model
        logger.info(f"OpenAI client initialised (model: {model})")

    def complete(
        self,
        prompt:      str,
        system:      str = SYSTEM_PROMPT,
        temperature: float = LLM_TEMPERATURE,
        max_tokens:  int   = LLM_MAX_TOKENS,
    ) -> str:
        messages = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            cost = estimate_cost(prompt + system, content, self._model)
            logger.debug(f"OpenAI call: ~{estimate_tokens(content)} tokens out | est. cost ${cost}")
            return content
        except Exception as exc:
            logger.error(f"OpenAI API error: {exc}")
            raise


# ══════════════════════════════════════════════════════════════════════════════
# Local GGUF Client (llama-cpp-python)
# ══════════════════════════════════════════════════════════════════════════════

class LocalLLMClient(LLMClient):
    """
    Runs a local GGUF model via llama-cpp-python.
    Download models from: https://huggingface.co/TheBloke (Mistral, Llama-3, etc.)
    Place the .gguf file at config.LOCAL_MODEL_PATH.
    """

    def __init__(
        self,
        model_path:  str   = LOCAL_MODEL_PATH,
        n_ctx:       int   = 4096,
        n_gpu_layers: int  = 0,     # set >0 to offload to GPU
        verbose:     bool  = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Local model not found: {model_path}\n"
                "Download a GGUF model (e.g., Mistral-7B-Instruct) and update LOCAL_MODEL_PATH in .env"
            )

        logger.info(f"Loading local LLM: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        logger.info("Local LLM loaded.")

    def complete(
        self,
        prompt:      str,
        system:      str = SYSTEM_PROMPT,
        temperature: float = LLM_TEMPERATURE,
        max_tokens:  int   = LLM_MAX_TOKENS,
    ) -> str:
        full_prompt = (
            f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )
        try:
            output = self._llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]"],
            )
            return output["choices"][0]["text"].strip()
        except Exception as exc:
            logger.error(f"Local LLM error: {exc}")
            raise


# ══════════════════════════════════════════════════════════════════════════════
# Mock Client (no API key / no model needed — for testing)
# ══════════════════════════════════════════════════════════════════════════════

class MockLLMClient(LLMClient):
    """
    Returns structured dummy responses.
    Use when no API key is available — lets the full pipeline run without LLM.
    """

    def complete(self, prompt: str, system: str = SYSTEM_PROMPT,
                 temperature: float = LLM_TEMPERATURE,
                 max_tokens: int = LLM_MAX_TOKENS) -> str:
        logger.info("[MockLLM] Returning placeholder response.")
        return json.dumps({
            "enriched_causes":              ["Instrument failure", "Operator error", "Corrosion"],
            "enriched_consequences":        ["Loss of containment", "Fire or explosion"],
            "enriched_safeguards_existing": ["High-pressure alarm", "PRV"],
            "enriched_safeguards_recommended": [
                "Install SIS interlock", "Upgrade corrosion monitoring", "Conduct LOPA"
            ],
            "severity": 4,
            "likelihood": 2,
            "historical_incidents": ["CSB Texas City 2005", "CSB T2 Laboratories 2007"],
            "narrative_notes": (
                "[MockLLM] This is a placeholder. Configure OPENAI_API_KEY or "
                "LOCAL_MODEL_PATH to get real LLM enrichment."
            ),
        })


# ══════════════════════════════════════════════════════════════════════════════
# Client Factory
# ══════════════════════════════════════════════════════════════════════════════

import re

def get_llm_client(provider: str = LLM_PROVIDER, mock_fallback: bool = True) -> LLMClient:
    """
    Factory: returns the appropriate LLM client based on config.
    Falls back to MockLLMClient if configured client fails to initialise.
    """
    try:
        if provider == "openai":
            return OpenAIClient()
        elif provider == "local":
            return LocalLLMClient()
        else:
            logger.warning(f"Unknown LLM provider '{provider}'. Using mock.")
            return MockLLMClient()
    except Exception as exc:
        if mock_fallback:
            logger.warning(f"LLM client init failed ({exc}). Falling back to MockLLMClient.")
            return MockLLMClient()
        raise


# ══════════════════════════════════════════════════════════════════════════════
# High-Level Functions
# ══════════════════════════════════════════════════════════════════════════════

def enrich_hazop_row(row_dict: dict, context: str, client: Optional[LLMClient] = None) -> dict:
    """
    Use LLM to enrich a single HAZOP row with deeper causes/consequences/safeguards.
    Returns the row_dict updated with LLM enrichment.
    """
    client = client or get_llm_client()

    # Truncate context to ~800 chars to stay within token budget
    ctx_snippet = context[:800].replace("\n", " ")

    prompt = HAZOP_ENRICHMENT_TEMPLATE.format(
        context=ctx_snippet,
        node_id=row_dict.get("node_id", ""),
        equipment=row_dict.get("equipment", ""),
        chemical=row_dict.get("chemical", ""),
        deviation=row_dict.get("deviation", ""),
        causes=", ".join(row_dict.get("causes", [])[:3]),
        consequences=", ".join(row_dict.get("consequences", [])[:2]),
        safeguards=", ".join(row_dict.get("safeguards_existing", [])[:3]),
    )

    enrichment = client.complete_json(prompt)

    if "parse_error" not in enrichment:
        # Merge enrichment into row
        row_dict["causes"]                  = enrichment.get("enriched_causes",              row_dict.get("causes", []))
        row_dict["consequences"]            = enrichment.get("enriched_consequences",         row_dict.get("consequences", []))
        row_dict["safeguards_existing"]     = enrichment.get("enriched_safeguards_existing",  row_dict.get("safeguards_existing", []))
        row_dict["safeguards_recommended"]  = enrichment.get("enriched_safeguards_recommended", row_dict.get("safeguards_recommended", []))
        row_dict["historical_ref"]          = "; ".join(enrichment.get("historical_incidents", [row_dict.get("historical_ref", "")]))
        row_dict["notes"]                   = enrichment.get("narrative_notes", row_dict.get("notes", ""))
        # Update risk if LLM gave values
        new_sev  = enrichment.get("severity",   None)
        new_like = enrichment.get("likelihood", None)
        if new_sev and new_like:
            from src.hazop_engine import rate_risk, priority_from_risk
            new_risk = rate_risk(new_sev, new_like)
            row_dict["risk"]            = new_risk.__dict__ if hasattr(new_risk, "__dict__") else asdict(new_risk)
            row_dict["action_priority"] = priority_from_risk(new_risk)

    row_dict["llm_enriched"] = True
    return row_dict


def enrich_all_rows(
    rows: list[dict],
    context: str,
    client: Optional[LLMClient] = None,
    max_rows: int = 20,
    delay: float = 0.5,
) -> list[dict]:
    """
    Enrich the top `max_rows` highest-risk rows with LLM.
    Rows are sorted by risk_score descending before enrichment.
    """
    client = client or get_llm_client()

    # Sort by risk score descending
    def _risk_score(r):
        risk = r.get("risk", {})
        if isinstance(risk, dict):
            return risk.get("risk_score", 0)
        return getattr(risk, "risk_score", 0)

    sorted_rows = sorted(rows, key=_risk_score, reverse=True)
    to_enrich   = sorted_rows[:max_rows]
    rest        = sorted_rows[max_rows:]

    enriched = []
    for i, row in enumerate(to_enrich, 1):
        logger.info(f"Enriching row {i}/{len(to_enrich)}: {row.get('deviation','')}")
        try:
            enriched.append(enrich_hazop_row(row, context, client))
        except Exception as exc:
            logger.warning(f"Row enrichment failed: {exc}")
            enriched.append(row)
        time.sleep(delay)

    return enriched + rest


def generate_summary(
    rows: list[dict],
    stats: dict,
    context: str,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Generate a natural-language executive summary of the HAZOP findings.
    """
    client = client or get_llm_client()

    # Select top 5 critical/high rows for summary
    def _risk_score(r):
        risk = r.get("risk", {})
        return risk.get("risk_score", 0) if isinstance(risk, dict) else 0

    top_rows = sorted(rows, key=_risk_score, reverse=True)[:5]
    top_rows_text = "\n".join(
        f"- {r.get('deviation','')} ({r.get('equipment','')} / {r.get('chemical','')}): "
        f"Risk={r.get('risk',{}).get('risk_level','?')}, "
        f"Causes={r.get('causes',['?'])[:2]}"
        for r in top_rows
    )

    prompt = SUMMARY_TEMPLATE.format(
        stats=json.dumps(stats, indent=2),
        top_rows=top_rows_text,
        context=context[:600].replace("\n", " "),
    )

    return client.complete(prompt)


def answer_question(question: str, context: str, client: Optional[LLMClient] = None) -> str:
    """
    Answer a specific question about an incident report using the LLM.
    """
    client = client or get_llm_client()
    prompt = QA_TEMPLATE.format(context=context[:2000], question=question)
    return client.complete(prompt)


def validate_ner(text: str, entities: list[dict], client: Optional[LLMClient] = None) -> dict:
    """
    Ask the LLM to validate and correct NER extractions.
    """
    client = client or get_llm_client()
    ents_text = "\n".join(
        f"  - '{e['text']}' → {e['label']}" for e in entities[:20]
    )
    prompt = NER_VALIDATION_TEMPLATE.format(
        text=text[:500],
        entities=ents_text,
    )
    return client.complete_json(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass helper (avoid circular import)
# ══════════════════════════════════════════════════════════════════════════════

def asdict(obj):
    """Minimal asdict for dataclass-like objects."""
    try:
        from dataclasses import asdict as dc_asdict
        return dc_asdict(obj)
    except Exception:
        return vars(obj)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_context = (
        "A hydrogen sulfide leak from a corroded pipeline at the refinery "
        "caused an explosion resulting in two fatalities. "
        "The pressure relief valve had failed to open due to corrosion. "
        "Recommended safeguard: install gas detectors and upgrade to corrosion-resistant PRVs."
    )

    print("Testing LLM Q&A (mock)...")
    client = MockLLMClient()
    answer = answer_question("What caused the explosion?", sample_context, client)
    print(f"Answer: {answer}\n")

    print("Testing LLM enrichment (mock)...")
    sample_row = {
        "node_id": "N001", "equipment": "Pipeline",
        "chemical": "Hydrogen Sulfide", "deviation": "No/None flow",
        "causes": ["Corrosion"], "consequences": ["Explosion"],
        "safeguards_existing": ["PRV"], "safeguards_recommended": [],
        "historical_ref": "", "notes": "",
    }
    enriched = enrich_hazop_row(sample_row, sample_context, client)
    print(f"Enriched causes: {enriched['causes']}")
    print(f"LLM notes: {enriched['notes']}")
