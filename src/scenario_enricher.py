"""
scenario_enricher.py — GenAI enrichment for compound failure scenarios.

Takes a ScenarioSet (produced by ScenarioGenerator) and enriches the
top-N highest-risk scenarios with:
  - Detailed causal mechanism narrative
  - Refined consequences (with severity justification)
  - Safeguard gap analysis
  - Specific actionable recommendations
  - Historical incident precedents (CSB / EPSC references)
  - Revised severity / likelihood ratings

Two enrichment strategies:
  A. Batch prompt  — sends multiple scenarios per LLM call (cheaper)
  B. Single prompt — one LLM call per scenario (more detailed)
"""

from __future__ import annotations

import json
import time
from typing import Optional
from loguru import logger

from src.scenario_models import CompoundScenario, ScenarioSet, RiskLevel
from src.llm_inference   import LLMClient, get_llm_client, MockLLMClient


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a world-class process safety engineer with 25+ years of experience
conducting HAZOP studies, incident investigations, and quantitative risk assessments for
chemical, petrochemical, and refining facilities.

You are familiar with:
- IEC 61882 HAZOP methodology
- CCPS Guidelines for Hazard Evaluation Procedures
- All major CSB accident investigations
- LOPA, QRA, bowtie analysis
- API 520/521 relief system design
- ISA-84 / IEC 61511 safety instrumented systems

When analysing compounding failure scenarios:
- Think like a HAZOP facilitator who has seen real accidents
- Consider both the immediate physical sequence AND the organisational/management failures
- Reference real historical accidents by name when they are genuinely similar
- Be specific: name the exact safeguards that fail, not just "safeguards fail"
- Quantify where possible: temperatures, pressures, flow rates
- NEVER speculate beyond what is physically plausible
"""

SINGLE_SCENARIO_PROMPT = """
Analyse the following compounding HAZOP failure scenario and provide a detailed expert assessment.

**Process System:** {system_name}
**Scenario ID:** {scenario_id}

**Failure Chain ({depth} simultaneous/sequential failures):**
{event_chain}

**Process Context:**
{context}

**Initial Assessment:**
- Preliminary consequences: {consequences}
- Identified safeguard gaps: {safeguard_gaps}
- Preliminary risk: Severity {severity}/5, Likelihood {likelihood}/5

Provide your expert analysis as JSON with this exact structure:
{{
  "headline": "One-line summary of the worst-case outcome (max 120 chars)",
  "mechanism": "2-3 sentence technical narrative of the causal chain — how each failure leads to the next",
  "consequences": [
    "Specific consequence 1 with physical detail",
    "Specific consequence 2",
    "Worst-case consequence 3"
  ],
  "top_consequence": "Single worst-case outcome in one sentence",
  "safeguard_gaps": [
    "Specific gap 1: which safeguard is defeated and why",
    "Specific gap 2"
  ],
  "existing_safeguards": [
    "Safeguard that remains effective in this scenario"
  ],
  "recommendations": [
    "Specific engineering recommendation 1",
    "Specific engineering recommendation 2",
    "Specific engineering recommendation 3"
  ],
  "historical_precedent": "Name of real incident this resembles, or 'No direct precedent found'",
  "severity": <1-5>,
  "likelihood": <1-5>,
  "severity_justification": "Why you rated severity as X",
  "likelihood_justification": "Why you rated likelihood as Y"
}}

Only output valid JSON. No markdown fences. Be specific and technically precise.
"""

BATCH_SCENARIO_PROMPT = """
You are reviewing {count} HAZOP compound failure scenarios for the process system: {system_name}

For each scenario, provide a brief expert enrichment.

**Scenarios:**
{scenarios_text}

Return a JSON array with one object per scenario:
[
  {{
    "scenario_id": "...",
    "headline": "One-line worst-case summary",
    "mechanism": "2-sentence causal narrative",
    "top_consequence": "Worst-case outcome",
    "recommendations": ["Rec 1", "Rec 2"],
    "historical_precedent": "Similar real accident or 'None'",
    "severity": <1-5>,
    "likelihood": <1-5>
  }}
]

Only output valid JSON array. No markdown.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Context builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_context(scenario: CompoundScenario, system_name: str) -> str:
    """Build process context string from scenario events."""
    lines = []
    seen_nodes = set()
    for ev in scenario.events:
        if ev.node_tag not in seen_nodes:
            seen_nodes.add(ev.node_tag)
            lines.append(f"  - {ev.node_tag} ({ev.node_type}): {ev.node_name}")
    return "\n".join(lines) or f"Equipment in {system_name}"


def _build_event_chain_text(scenario: CompoundScenario) -> str:
    lines = []
    for i, ev in enumerate(scenario.events, 1):
        lines.append(
            f"  {i}. [{ev.node_tag}] {ev.failure_mode.value if hasattr(ev.failure_mode, 'value') else ev.failure_mode}"
            f" → {ev.guide_word} {ev.parameter}\n"
            f"     {ev.description}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Single-scenario enrichment
# ══════════════════════════════════════════════════════════════════════════════

def enrich_scenario(
    scenario: CompoundScenario,
    system_name: str,
    client: LLMClient,
) -> CompoundScenario:
    """Enrich one scenario with a dedicated LLM call."""
    prompt = SINGLE_SCENARIO_PROMPT.format(
        system_name=system_name,
        scenario_id=scenario.short_id(),
        depth=scenario.compound_depth,
        event_chain=_build_event_chain_text(scenario),
        context=_build_context(scenario, system_name),
        consequences="; ".join(scenario.consequences[:3]),
        safeguard_gaps="; ".join(scenario.safeguard_gaps[:3]) or "None identified",
        severity=scenario.severity,
        likelihood=scenario.likelihood,
    )

    result = client.complete_json(prompt, system=SYSTEM_PROMPT)

    if "parse_error" not in result:
        scenario.headline     = result.get("headline",     scenario.headline)
        scenario.mechanism    = result.get("mechanism",    "")
        scenario.consequences = result.get("consequences", scenario.consequences)
        scenario.top_consequence = result.get("top_consequence", "")
        scenario.safeguard_gaps  = result.get("safeguard_gaps",  scenario.safeguard_gaps)
        scenario.existing_safeguards = result.get("existing_safeguards", [])
        scenario.recommendations     = result.get("recommendations", [])
        scenario.historical_precedent = result.get("historical_precedent", "")

        new_sev = result.get("severity",   scenario.severity)
        new_lk  = result.get("likelihood", scenario.likelihood)
        if isinstance(new_sev, int) and isinstance(new_lk, int):
            scenario.severity   = max(1, min(5, new_sev))
            scenario.likelihood = max(1, min(5, new_lk))
            scenario.recompute_risk()

    scenario.llm_enriched = True
    return scenario


# ══════════════════════════════════════════════════════════════════════════════
# Batch enrichment (cheaper — multiple scenarios per call)
# ══════════════════════════════════════════════════════════════════════════════

def _format_scenario_for_batch(scenario: CompoundScenario) -> str:
    return (
        f"ID: {scenario.short_id()}\n"
        f"Chain: {scenario.event_chain_text()}\n"
        f"Current risk: S{scenario.severity}/L{scenario.likelihood}"
    )


def enrich_batch(
    scenarios: list[CompoundScenario],
    system_name: str,
    client: LLMClient,
    batch_size: int = 5,
) -> list[CompoundScenario]:
    """Enrich multiple scenarios per LLM call. Faster and cheaper."""
    id_map = {sc.short_id(): sc for sc in scenarios}
    enriched_ids = set()

    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        scenarios_text = "\n\n---\n\n".join(
            _format_scenario_for_batch(sc) for sc in batch
        )
        prompt = BATCH_SCENARIO_PROMPT.format(
            count=len(batch),
            system_name=system_name,
            scenarios_text=scenarios_text,
        )
        try:
            raw = client.complete(prompt, system=SYSTEM_PROMPT, max_tokens=2000)
            import re
            raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
            raw = re.sub(r'\n?```$', '', raw)
            results = json.loads(raw)
            if not isinstance(results, list):
                results = [results]
        except Exception as exc:
            logger.warning(f"Batch enrichment parse error: {exc}")
            continue

        for res in results:
            sc_id = res.get("scenario_id", "")
            sc    = id_map.get(sc_id)
            if not sc:
                continue
            sc.headline           = res.get("headline",    sc.headline)
            sc.mechanism          = res.get("mechanism",   "")
            sc.top_consequence    = res.get("top_consequence", "")
            sc.recommendations    = res.get("recommendations", [])
            sc.historical_precedent = res.get("historical_precedent", "")
            new_sev = res.get("severity",   sc.severity)
            new_lk  = res.get("likelihood", sc.likelihood)
            if isinstance(new_sev, int) and isinstance(new_lk, int):
                sc.severity   = max(1, min(5, new_sev))
                sc.likelihood = max(1, min(5, new_lk))
                sc.recompute_risk()
            sc.llm_enriched = True
            enriched_ids.add(sc_id)

    logger.info(f"Batch enrichment: {len(enriched_ids)}/{len(scenarios)} enriched")
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# Main enrichment orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioEnricher:
    """
    Orchestrates LLM enrichment of a ScenarioSet.

    Strategy:
      - Top `deep_enrich_n` scenarios → single dedicated LLM call each (most detail)
      - Next `batch_enrich_n` scenarios → batch LLM calls (faster)
      - Remaining → left with rule-based content only
    """

    def __init__(
        self,
        client:          Optional[LLMClient] = None,
        deep_enrich_n:   int = 10,
        batch_enrich_n:  int = 50,
        batch_size:      int = 5,
        call_delay:      float = 0.3,
        provider:        str = "mock",
    ):
        self.client         = client or get_llm_client(provider)
        self.deep_enrich_n  = deep_enrich_n
        self.batch_enrich_n = batch_enrich_n
        self.batch_size     = batch_size
        self.call_delay     = call_delay

    def enrich(self, scenario_set: ScenarioSet) -> ScenarioSet:
        """
        Enrich scenarios in-place. Returns the same ScenarioSet with
        llm_enriched=True on all processed scenarios.
        """
        scenarios    = scenario_set.scenarios
        system_name  = scenario_set.system_name

        # Sort by risk descending (already should be, but ensure)
        scenarios.sort(key=lambda s: (-s.risk_score, -s.compound_depth))

        # Tier 1: deep single enrichment
        tier1 = scenarios[:self.deep_enrich_n]
        tier2 = scenarios[self.deep_enrich_n:self.deep_enrich_n + self.batch_enrich_n]

        logger.info(
            f"Enriching: {len(tier1)} deep + {len(tier2)} batch "
            f"of {len(scenarios)} total scenarios"
        )

        for i, sc in enumerate(tier1, 1):
            logger.info(f"Deep enriching {i}/{len(tier1)}: {sc.short_id()} — {sc.headline}")
            try:
                enrich_scenario(sc, system_name, self.client)
            except Exception as exc:
                logger.warning(f"Deep enrichment failed for {sc.short_id()}: {exc}")
            time.sleep(self.call_delay)

        if tier2:
            logger.info(f"Batch enriching {len(tier2)} scenarios…")
            enrich_batch(tier2, system_name, self.client, self.batch_size)

        scenario_set.compute_stats()
        return scenario_set

    def enrich_single(self, scenario: CompoundScenario,
                      system_name: str) -> CompoundScenario:
        """Enrich a single scenario on demand (used in UI for drill-down)."""
        return enrich_scenario(scenario, system_name, self.client)


# ══════════════════════════════════════════════════════════════════════════════
# Mock enrichment (for offline / testing)
# ══════════════════════════════════════════════════════════════════════════════

MOCK_PRECEDENTS = [
    "CSB Texas City Refinery 2005 — multiple simultaneous failures",
    "CSB T2 Laboratories 2007 — cooling failure + runaway reaction",
    "CSB Bhopal 1984 — loss of cooling + contamination",
    "CSB Caribbean Petroleum 2009 — tank overfill + ignition",
    "CSB West Fertilizer 2013 — fire + reactive chemical decomposition",
    "Piper Alpha 1988 — permit-to-work failure + gas release + ignition",
    "Flixborough 1974 — temporary bypass pipe failure",
    "Texas City 1947 — ammonium nitrate detonation",
]

import random

def mock_enrich_scenario(scenario: CompoundScenario) -> CompoundScenario:
    """Quick mock enrichment for testing — no LLM needed."""
    depth = scenario.compound_depth
    chems = list({
        ev.node_name for ev in scenario.events
    })

    scenario.mechanism = (
        f"The {scenario.events[0].failure_mode.value.lower()} on "
        f"{scenario.events[0].node_tag} initiates a cascade: "
        + (f"this directly causes {scenario.events[1].failure_mode.value.lower()} "
           f"on {scenario.events[1].node_tag}, " if depth > 1 else "")
        + "defeating the primary safeguard layer and escalating to the identified consequences."
    )
    scenario.top_consequence = (
        scenario.consequences[0] if scenario.consequences
        else f"Major process safety event involving {', '.join(chems[:2])}"
    )
    scenario.existing_safeguards = [
        f"Independent alarm on {scenario.events[0].node_tag}",
        "Operator response to initial indication",
    ]
    scenario.recommendations = [
        f"Install independent high-integrity protection on {scenario.events[0].node_tag}",
        f"Conduct LOPA for {scenario.headline[:60]} scenario",
        "Review safeguard independence — ensure no common-cause defeat",
    ]
    scenario.historical_precedent = random.choice(MOCK_PRECEDENTS)
    scenario.llm_enriched = True
    return scenario


def mock_enrich_set(scenario_set: ScenarioSet, top_n: int = 50) -> ScenarioSet:
    """Apply mock enrichment to top_n scenarios."""
    for sc in scenario_set.scenarios[:top_n]:
        mock_enrich_scenario(sc)
    scenario_set.compute_stats()
    return scenario_set
