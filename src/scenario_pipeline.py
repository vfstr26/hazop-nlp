"""
scenario_pipeline.py — End-to-end HAZOP Scenario Generation Pipeline.

Wires together:
  pid_parser → scenario_generator → scenario_enricher → scenario_ranker
  → output_formatter (JSON / HTML / Excel / CSV)

Also bridges into the existing hazop_nlp pipeline so NER-extracted
entities from incident reports can seed the P&ID parser.

Public API:
  run_pipeline(input_data, ...)  → ScenarioSet
  export_scenario_set(...)       → dict of file paths
  scenarios_to_hazop_rows(...)   → list[dict] for output_formatter
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from loguru import logger

from src.scenario_models    import ScenarioSet, CompoundScenario
from src.pid_parser         import parse, build_sample_pid, PIDSystem
from src.scenario_generator import ScenarioGenerator
from src.scenario_enricher  import ScenarioEnricher, mock_enrich_set
from src.scenario_ranker    import ScenarioRanker, get_statistics
from config                 import OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    def __init__(
        self,
        # Generator
        max_depth:      int   = 3,
        max_scenarios:  int   = 2000,
        min_risk_score: int   = 2,
        # Enricher
        llm_provider:   str   = "mock",
        deep_enrich_n:  int   = 10,
        batch_enrich_n: int   = 40,
        # Ranker
        top_n:          int   = 200,
        keep_all_critical: bool = True,
        # Parser
        use_llm_parser: bool  = False,
    ):
        self.max_depth       = max_depth
        self.max_scenarios   = max_scenarios
        self.min_risk_score  = min_risk_score
        self.llm_provider    = llm_provider
        self.deep_enrich_n   = deep_enrich_n
        self.batch_enrich_n  = batch_enrich_n
        self.top_n           = top_n
        self.keep_all_critical = keep_all_critical
        self.use_llm_parser  = use_llm_parser


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_data: Union[str, dict, Path],
    config: Optional[PipelineConfig] = None,
    llm_client=None,
    system_name: str = "Process System",
    progress_callback=None,    # callable(step: str, pct: int)
) -> ScenarioSet:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    input_data      : P&ID description (str), JSON dict, file Path, or PIDSystem
    config          : PipelineConfig (defaults used if None)
    llm_client      : LLMClient instance (auto-created if None)
    system_name     : Display name for the process system
    progress_callback: Optional callable(step_name, pct_complete) for UI updates

    Returns
    -------
    ScenarioSet with generated, enriched, and ranked scenarios
    """
    cfg = config or PipelineConfig()

    def _progress(step: str, pct: int):
        if progress_callback:
            progress_callback(step, pct)
        logger.info(f"[{pct:3d}%] {step}")

    # ── Step 1: Parse P&ID ────────────────────────────────────────────────────
    _progress("Parsing P&ID description…", 5)

    if isinstance(input_data, PIDSystem):
        pid_system = input_data
    else:
        if llm_client and cfg.use_llm_parser:
            from src.pid_parser import parse_text_with_llm
            pid_system = parse_text_with_llm(
                str(input_data), llm_client, system_name
            )
        else:
            pid_system = parse(input_data, system_name=system_name)

    if not pid_system.nodes:
        logger.warning("No nodes found in P&ID — using sample system")
        pid_system = build_sample_pid()

    _progress(f"P&ID parsed: {len(pid_system.nodes)} nodes, "
              f"{len(pid_system.streams)} streams", 15)

    # ── Step 2: Generate scenarios ────────────────────────────────────────────
    _progress("Generating compound failure scenarios…", 20)

    generator = ScenarioGenerator(
        max_depth=cfg.max_depth,
        max_scenarios=cfg.max_scenarios,
        min_risk_score=cfg.min_risk_score,
    )
    scenario_set = generator.run(pid_system)

    _progress(f"Generated {scenario_set.total} raw scenarios", 50)

    # ── Step 3: Rank & deduplicate ────────────────────────────────────────────
    _progress("Ranking and deduplicating…", 55)

    ranker = ScenarioRanker(
        top_n=cfg.top_n,
        keep_all_critical=cfg.keep_all_critical,
    )
    scenario_set = ranker.rank(scenario_set)

    _progress(f"Ranked to top {len(scenario_set.scenarios)} scenarios", 65)

    # ── Step 4: LLM enrichment ─────────────────────────────────────────────────
    _progress("Enriching with GenAI analysis…", 70)

    if cfg.llm_provider == "mock" or llm_client is None:
        mock_enrich_set(scenario_set, top_n=min(50, len(scenario_set.scenarios)))
        _progress("Mock enrichment applied", 90)
    else:
        enricher = ScenarioEnricher(
            client=llm_client,
            deep_enrich_n=cfg.deep_enrich_n,
            batch_enrich_n=cfg.batch_enrich_n,
            provider=cfg.llm_provider,
        )
        scenario_set = enricher.enrich(scenario_set)
        _progress("LLM enrichment complete", 90)

    # ── Step 5: Final stats ───────────────────────────────────────────────────
    scenario_set.compute_stats()
    _progress("Pipeline complete ✓", 100)

    logger.info(
        f"Pipeline done: {scenario_set.total} scenarios | "
        + " | ".join(f"{k}={v}" for k, v in scenario_set.by_risk.items())
    )
    return scenario_set


# ══════════════════════════════════════════════════════════════════════════════
# Bridge: NER result → seeded P&ID
# ══════════════════════════════════════════════════════════════════════════════

def pid_from_ner(ner_result: dict, system_name: str = "NER-derived System") -> PIDSystem:
    """
    Build a PIDSystem from the NER extraction results of an incident report.
    Used when the user provides an incident text rather than a P&ID description.
    """
    from src.pid_parser import PIDSystem, PIDNode, PIDStream, DesignCondition, NodeType

    unique = ner_result.get("unique_terms", {})
    equips = unique.get("EQUIP", [])
    chems  = unique.get("CHEM",  [])
    meta   = ner_result.get("metadata", {})

    system = PIDSystem(
        name=system_name,
        description=f"Derived from incident: {meta.get('source', '')}",
        utilities=["cooling water", "steam", "instrument air"],
    )

    # One node per equipment entity
    for i, eq in enumerate(equips[:10]):
        node_type = _guess_node_type(eq)
        node = PIDNode(
            tag=f"E-{i+1:03d}",
            name=eq.title(),
            node_type=node_type,
            chemicals=[c.title() for c in chems[:3]],
            conditions=DesignCondition(),
            safeguards=list({
                sg.title()
                for sg in unique.get("SAFEGUARD", [])[:4]
            }),
            description=f"Extracted from incident report: {meta.get('source', '')}",
        )
        system.nodes.append(node)

    # Linear stream chain
    for i in range(len(system.nodes) - 1):
        system.streams.append(PIDStream(
            stream_id=f"S-{i+1:03d}",
            from_tag=system.nodes[i].tag,
            to_tag=system.nodes[i + 1].tag,
            chemicals=[c.title() for c in chems[:2]],
        ))

    return system


def _guess_node_type(name: str):
    from src.pid_parser import _detect_node_type
    return _detect_node_type(name)


# ══════════════════════════════════════════════════════════════════════════════
# Bridge: CompoundScenario → HAZOP row dict (for output_formatter)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_to_hazop_row(scenario: CompoundScenario, node_counter: int = 1) -> dict:
    """Convert a CompoundScenario to the hazop_engine row format for output_formatter."""
    risk = {
        "risk_level":       scenario.risk_level.value,
        "risk_score":       scenario.risk_score,
        "severity":         scenario.severity,
        "likelihood":       scenario.likelihood,
        "severity_desc":    f"Severity {scenario.severity}/5",
        "likelihood_desc":  f"Likelihood {scenario.likelihood}/5",
    }
    nodes_involved = list({ev.node_tag for ev in scenario.events})
    chems_involved = list({
        c for ev in scenario.events
        for c in ([] if not ev.node_name else [])
    })

    return {
        "node_id":        f"SC-{node_counter:04d}",
        "equipment":      ", ".join(nodes_involved),
        "chemical":       ", ".join(chems_involved) or "Process fluid",
        "parameter":      scenario.events[0].parameter if scenario.events else "",
        "guide_word":     scenario.events[0].guide_word if scenario.events else "",
        "deviation":      scenario.headline or scenario.events[0].description if scenario.events else "",
        "causes":         [ev.description for ev in scenario.events],
        "consequences":   scenario.consequences,
        "safeguards_existing":    scenario.existing_safeguards,
        "safeguards_recommended": scenario.recommendations,
        "risk":           risk,
        "actions":        scenario.recommendations[:3],
        "action_priority":scenario.action_priority,
        "historical_ref": scenario.historical_precedent,
        "notes":          scenario.mechanism,
        "compound_depth": scenario.compound_depth,
        "scenario_id":    scenario.short_id(),
        "llm_enriched":   scenario.llm_enriched,
    }


def scenarios_to_hazop_rows(scenario_set: ScenarioSet) -> list[dict]:
    """Convert all scenarios in a set to HAZOP row dicts."""
    return [
        scenario_to_hazop_row(sc, i + 1)
        for i, sc in enumerate(scenario_set.scenarios)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export_scenario_set(
    scenario_set: ScenarioSet,
    output_dir: Optional[Path] = None,
    formats: list[str] = None,
) -> dict[str, Path]:
    """
    Export scenario set in multiple formats using output_formatter.
    Returns dict of format → Path.
    """
    from src.output_formatter import export_all

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json", "html", "excel", "csv"]

    rows    = scenarios_to_hazop_rows(scenario_set)
    stats   = get_statistics(scenario_set)
    summary = {
        "total_rows":          len(rows),
        "unique_nodes":         len(scenario_set.by_node),
        "unique_deviations":    len({r["deviation"] for r in rows}),
        "risk_breakdown":       scenario_set.by_risk,
        "priority_breakdown": {
            "Immediate":   sum(1 for r in rows if r.get("action_priority") == "Immediate"),
            "Short-term":  sum(1 for r in rows if r.get("action_priority") == "Short-term"),
            "Long-term":   sum(1 for r in rows if r.get("action_priority") == "Long-term"),
        },
    }
    meta = {
        "source":     scenario_set.system_name,
        "doc_id":     scenario_set.set_id,
        "date":       scenario_set.generated_at[:10],
        "chemicals":  [],
        "generator":  "HAZOP Scenario Generator v1.0",
        "stats":      stats,
    }

    # Also save the raw ScenarioSet JSON
    raw_path = output_dir / f"{scenario_set.set_id}_scenarios.json"
    scenario_set.save(raw_path)
    logger.info(f"Raw ScenarioSet → {raw_path}")

    paths = export_all(rows, summary, meta,
                       doc_id=f"scenarios_{scenario_set.set_id}",
                       output_dir=output_dir)
    paths["scenarios_raw"] = raw_path
    return paths
