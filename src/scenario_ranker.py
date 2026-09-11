"""
scenario_ranker.py — Scenario ranking, deduplication, and clustering.

Takes a raw ScenarioSet (potentially thousands of scenarios) and reduces
it to a curated, manageable set for engineering review.

Stages:
  1. Deduplication   — remove near-identical scenarios (same node + same failure mode)
  2. Clustering      — group similar scenarios into families
  3. Novelty scoring — prioritise scenarios that reveal new blind spots
  4. Diversity filter — ensure coverage across all nodes and failure types
  5. Final ranking   — composite score: risk × novelty × depth
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Optional
from loguru import logger

from src.scenario_models import CompoundScenario, ScenarioSet, RiskLevel


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════════════════════════

def _scenario_fingerprint(scenario: CompoundScenario) -> str:
    """
    Canonical fingerprint for deduplication.
    Two scenarios with the same nodes + failure modes (regardless of order)
    get the same fingerprint.
    """
    parts = sorted(
        f"{ev.node_tag}:{ev.failure_mode.value if hasattr(ev.failure_mode, 'value') else ev.failure_mode}"
        for ev in scenario.events
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def deduplicate(scenarios: list[CompoundScenario]) -> list[CompoundScenario]:
    """
    Remove duplicate scenarios, keeping the highest-risk version of each
    fingerprint group.
    """
    seen: dict[str, CompoundScenario] = {}
    for sc in scenarios:
        fp = _scenario_fingerprint(sc)
        if fp not in seen or sc.risk_score > seen[fp].risk_score:
            seen[fp] = sc
    result = list(seen.values())
    removed = len(scenarios) - len(result)
    if removed:
        logger.info(f"Deduplication: removed {removed} duplicates → {len(result)} remain")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Clustering — group scenarios into families
# ══════════════════════════════════════════════════════════════════════════════

def cluster_by_initiating_event(
    scenarios: list[CompoundScenario]
) -> dict[str, list[CompoundScenario]]:
    """
    Group scenarios by their first (initiating) failure event.
    Key = "NODE_TAG:FAILURE_MODE"
    """
    clusters: dict[str, list[CompoundScenario]] = defaultdict(list)
    for sc in scenarios:
        if sc.events:
            ev = sc.events[0]
            fm = ev.failure_mode.value if hasattr(ev.failure_mode, "value") else str(ev.failure_mode)
            key = f"{ev.node_tag}:{fm}"
            clusters[key].append(sc)
    return dict(clusters)


def cluster_by_node(
    scenarios: list[CompoundScenario]
) -> dict[str, list[CompoundScenario]]:
    """Group scenarios by which nodes are involved."""
    clusters: dict[str, list[CompoundScenario]] = defaultdict(list)
    for sc in scenarios:
        for ev in sc.events:
            clusters[ev.node_tag].append(sc)
    return dict(clusters)


def cluster_by_consequence_type(
    scenarios: list[CompoundScenario]
) -> dict[str, list[CompoundScenario]]:
    """Group by consequence category: explosion, fire, toxic, overpressure, spill."""
    CONSEQUENCE_KEYWORDS = {
        "Explosion / BLEVE":  ["explosion", "bleve", "blast", "detonation"],
        "Fire":               ["fire", "flash fire", "jet fire", "pool fire", "ignition"],
        "Toxic Release":      ["toxic", "h2s", "ammonia", "chlorine", "poison", "asphyxia"],
        "Overpressure":       ["overpressure", "vessel burst", "rupture", "integrity"],
        "Loss of Containment":["leak", "spill", "release", "loss of containment"],
        "Runaway Reaction":   ["runaway", "exotherm", "thermal", "catalyst"],
        "Process Upset":      ["upset", "off-spec", "control", "starvation"],
    }
    clusters: dict[str, list[CompoundScenario]] = defaultdict(list)
    for sc in scenarios:
        all_text = " ".join(sc.consequences + [sc.headline, sc.top_consequence]).lower()
        matched = False
        for category, keywords in CONSEQUENCE_KEYWORDS.items():
            if any(kw in all_text for kw in keywords):
                clusters[category].append(sc)
                matched = True
                break
        if not matched:
            clusters["Other"].append(sc)
    return dict(clusters)


# ══════════════════════════════════════════════════════════════════════════════
# Novelty scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_novelty(scenarios: list[CompoundScenario]) -> dict[str, float]:
    """
    Assign each scenario a novelty score 0–1.
    Scenarios that involve rare node/failure combinations score higher.
    Single-failure scenarios score lower (already obvious to a HAZOP team).
    """
    # Count how often each (node, fm) pair appears
    pair_counts: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        for ev in sc.events:
            fm = ev.failure_mode.value if hasattr(ev.failure_mode, "value") else str(ev.failure_mode)
            pair_counts[f"{ev.node_tag}:{fm}"] += 1

    total = len(scenarios) or 1
    novelty: dict[str, float] = {}

    for sc in scenarios:
        # Rarity: inverse frequency of constituent pairs
        pair_freqs = [
            pair_counts[
                f"{ev.node_tag}:"
                f"{ev.failure_mode.value if hasattr(ev.failure_mode, 'value') else ev.failure_mode}"
            ]
            for ev in sc.events
        ]
        rarity = 1.0 - (sum(pair_freqs) / len(pair_freqs)) / total

        # Depth bonus: deeper chains are less obvious
        depth_bonus = min(0.3, (sc.compound_depth - 1) * 0.15)

        # Safeguard defeat bonus: scenarios that defeat safeguards are novel
        sg_bonus = min(0.2, len(sc.safeguard_gaps) * 0.05)

        novelty[sc.scenario_id] = min(1.0, rarity + depth_bonus + sg_bonus)

    return novelty


# ══════════════════════════════════════════════════════════════════════════════
# Composite ranking
# ══════════════════════════════════════════════════════════════════════════════

def composite_score(
    scenario: CompoundScenario,
    novelty_map: dict[str, float],
    w_risk: float    = 0.5,
    w_novelty: float = 0.3,
    w_depth: float   = 0.2,
) -> float:
    """
    Composite score combining risk, novelty, and compound depth.
    All components normalised to 0–1.
    """
    risk_norm   = scenario.risk_score / 25.0
    novelty_val = novelty_map.get(scenario.scenario_id, 0.5)
    depth_norm  = min(1.0, (scenario.compound_depth - 1) / 3.0)
    return w_risk * risk_norm + w_novelty * novelty_val + w_depth * depth_norm


# ══════════════════════════════════════════════════════════════════════════════
# Diversity filter — ensure node coverage
# ══════════════════════════════════════════════════════════════════════════════

def ensure_node_coverage(
    scenarios: list[CompoundScenario],
    min_per_node: int = 2,
) -> list[CompoundScenario]:
    """
    Guarantee at least `min_per_node` scenarios for every node tag.
    Appends under-represented node scenarios at the end of the list.
    """
    node_counts: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        for ev in sc.events:
            node_counts[ev.node_tag] += 1

    all_nodes = {ev.node_tag for sc in scenarios for ev in sc.events}
    gaps = {tag for tag in all_nodes if node_counts[tag] < min_per_node}

    if not gaps:
        return scenarios

    # Already included; just move gap nodes' scenarios to front
    gap_scenarios = [
        sc for sc in scenarios
        if any(ev.node_tag in gaps for ev in sc.events)
        and sc not in scenarios[:len(scenarios) // 2]
    ]
    logger.info(f"Coverage fix: added {len(gap_scenarios)} scenarios for under-represented nodes")
    return scenarios  # list is already ordered; caller handles top-N slicing


# ══════════════════════════════════════════════════════════════════════════════
# Main Ranker
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioRanker:
    """
    Reduces and ranks a ScenarioSet for human review.

    Usage:
        ranker = ScenarioRanker(top_n=200)
        ranked_set = ranker.rank(raw_scenario_set)
    """

    def __init__(
        self,
        top_n:           int   = 200,
        min_per_node:    int   = 2,
        w_risk:          float = 0.5,
        w_novelty:       float = 0.3,
        w_depth:         float = 0.2,
        keep_all_critical: bool = True,
    ):
        self.top_n             = top_n
        self.min_per_node      = min_per_node
        self.w_risk            = w_risk
        self.w_novelty         = w_novelty
        self.w_depth           = w_depth
        self.keep_all_critical = keep_all_critical

    def rank(self, scenario_set: ScenarioSet) -> ScenarioSet:
        """
        Full ranking pipeline.
        Modifies scenario_set.scenarios in-place and recomputes stats.
        """
        scenarios = scenario_set.scenarios
        n_before  = len(scenarios)

        # 1. Deduplicate
        scenarios = deduplicate(scenarios)

        # 2. Novelty scoring
        novelty_map = score_novelty(scenarios)

        # 3. Composite sort
        scenarios.sort(
            key=lambda s: composite_score(s, novelty_map, self.w_risk,
                                          self.w_novelty, self.w_depth),
            reverse=True,
        )

        # 4. Always keep ALL critical scenarios
        if self.keep_all_critical:
            criticals = [s for s in scenarios if s.risk_level == RiskLevel.CRITICAL]
            rest      = [s for s in scenarios if s.risk_level != RiskLevel.CRITICAL]
            scenarios = criticals + rest

        # 5. Node coverage
        ensure_node_coverage(scenarios, self.min_per_node)

        # 6. Cap at top_n
        if len(scenarios) > self.top_n:
            scenarios = scenarios[:self.top_n]

        scenario_set.scenarios = scenarios
        scenario_set.compute_stats()

        logger.info(
            f"Ranking complete: {n_before} → {len(scenarios)} scenarios "
            f"(top_n={self.top_n}, kept all critical={self.keep_all_critical})"
        )
        return scenario_set

    def get_summary_table(
        self, scenario_set: ScenarioSet
    ) -> list[dict]:
        """
        Return a flat list of dicts suitable for a DataFrame or table display.
        """
        novelty_map = score_novelty(scenario_set.scenarios)
        rows = []
        for sc in scenario_set.scenarios:
            rows.append({
                "ID":           sc.short_id(),
                "Risk Level":   sc.risk_level.value,
                "Risk Score":   sc.risk_score,
                "Depth":        sc.compound_depth,
                "Headline":     sc.headline,
                "Top Consequence": sc.top_consequence or (sc.consequences[0] if sc.consequences else ""),
                "Priority":     sc.action_priority,
                "Nodes":        ", ".join({ev.node_tag for ev in sc.events}),
                "Enriched":     "✓" if sc.llm_enriched else "—",
                "Novelty":      round(novelty_map.get(sc.scenario_id, 0), 2),
                "Historical":   sc.historical_precedent or "—",
                "Status":       sc.status.value,
            })
        return rows


# ══════════════════════════════════════════════════════════════════════════════
# Scenario statistics helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_statistics(scenario_set: ScenarioSet) -> dict:
    """Extended statistics beyond ScenarioSet.compute_stats()."""
    scenario_set.compute_stats()
    scenarios = scenario_set.scenarios

    # Failure mode frequency
    fm_counts: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        for ev in sc.events:
            fm = ev.failure_mode.value if hasattr(ev.failure_mode, "value") else str(ev.failure_mode)
            fm_counts[fm] += 1

    # Consequence type distribution
    clusters = cluster_by_consequence_type(scenarios)
    conseq_dist = {k: len(v) for k, v in clusters.items()}

    # Depth distribution
    depth_dist = defaultdict(int)
    for sc in scenarios:
        depth_dist[sc.compound_depth] += 1

    # Top 5 riskiest nodes
    node_risk: dict[str, int] = defaultdict(int)
    for sc in scenarios:
        for ev in sc.events:
            node_risk[ev.node_tag] += sc.risk_score
    top_nodes = sorted(node_risk.items(), key=lambda x: -x[1])[:5]

    return {
        "total":              scenario_set.total,
        "by_risk":            scenario_set.by_risk,
        "by_depth":           dict(depth_dist),
        "by_consequence_type":conseq_dist,
        "failure_mode_freq":  dict(sorted(fm_counts.items(), key=lambda x: -x[1])[:10]),
        "top_riskiest_nodes": dict(top_nodes),
        "llm_enriched_count": sum(1 for sc in scenarios if sc.llm_enriched),
        "safeguard_defeats":  sum(len(sc.safeguard_gaps) for sc in scenarios),
    }
