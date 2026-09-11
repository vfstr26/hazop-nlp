"""
scenario_generator.py — Compounding Failure Scenario Generator.

Core idea:
  Every piece of equipment has a set of credible failure modes.
  When two or more failures occur together (or in sequence), they compound —
  defeating safeguards, accelerating consequences, creating blind spots that
  a tired HAZOP team would never brainstorm in a room.

Algorithm:
  1. For each node, enumerate all applicable FailureEvents (single failures)
  2. Generate compound chains:
       depth-1  : single equipment failures            → N scenarios
       depth-2  : pairs that are causally connected    → N² / 2 scenarios
       depth-3  : triples that escalate each other     → N³ / 6 scenarios
  3. Apply causal filters — prune physically impossible chains
  4. Apply safeguard-defeat rules — flag scenarios that defeat known safeguards
  5. Score initial risk (before LLM enrichment)

Produces potentially thousands of scenarios for a realistic P&ID.
The ranking engine then reduces this to a manageable set.
"""

from __future__ import annotations

import uuid
import itertools
from typing import Optional
from loguru import logger

from src.scenario_models import (
    PIDSystem, PIDNode, PIDStream,
    FailureEvent, FailureMode, CompoundScenario, ScenarioSet,
    NodeType, RiskLevel,
)


# ══════════════════════════════════════════════════════════════════════════════
# Failure Mode Library
# Maps NodeType → list of applicable (FailureMode, parameter, guide_word, likelihood)
# ══════════════════════════════════════════════════════════════════════════════

NODE_FAILURE_MODES: dict[NodeType, list[tuple[FailureMode, str, str, int]]] = {
    NodeType.REACTOR: [
        (FailureMode.LOSES_POWER,      "cooling",        "No/None",   3),
        (FailureMode.OVERHEATS,        "temperature",    "More",      3),
        (FailureMode.FAILS_HIGH,       "pressure",       "More",      3),
        (FailureMode.CATALYST_DEGRADE, "reaction rate",  "Less",      2),
        (FailureMode.FOULING,          "heat transfer",  "Less",      2),
        (FailureMode.LEAKS,            "containment",    "No/None",   2),
        (FailureMode.RUPTURES,         "containment",    "No/None",   1),
        (FailureMode.HUMAN_ERROR,      "procedure",      "Other Than",2),
        (FailureMode.WRONG_SIGNAL,     "temperature",    "More",      2),
    ],
    NodeType.VESSEL: [
        (FailureMode.FAILS_HIGH,       "level",          "More",      3),
        (FailureMode.FAILS_LOW,        "level",          "No/None",   3),
        (FailureMode.FAILS_HIGH,       "pressure",       "More",      2),
        (FailureMode.CORRODES,         "wall thickness", "Less", 2),
        (FailureMode.LEAKS,            "containment",    "No/None",   2),
        (FailureMode.RUPTURES,         "containment",    "No/None",   1),
        (FailureMode.FOULING,          "outlet",         "No/None",   2),
        (FailureMode.HUMAN_ERROR,      "procedure",      "Other Than",2),
    ],
    NodeType.HEAT_EXCHANGER: [
        (FailureMode.LOSES_POWER,      "cooling",        "No/None",   3),
        (FailureMode.FOULING,          "heat transfer",  "Less",      3),
        (FailureMode.LEAKS,            "tube bundle",    "As Well As",2),
        (FailureMode.CORRODES,         "tube bundle",    "Less",      2),
        (FailureMode.FAILS_CLOSED,     "cooling flow",   "No/None",   2),
        (FailureMode.FAILS_HIGH,       "temperature",    "More",      2),
        (FailureMode.VIBRATES,         "mechanical",     "Other Than",1),
    ],
    NodeType.PUMP: [
        (FailureMode.FAILS_CLOSED,     "flow",           "No/None",   3),
        (FailureMode.CAVITATION,       "flow",           "Less",      3),
        (FailureMode.LEAKS,            "seal",           "As Well As",2),
        (FailureMode.REVERSE_FLOW,     "flow",           "Reverse",   2),
        (FailureMode.LOSES_POWER,      "flow",           "No/None",   2),
        (FailureMode.VIBRATES,         "mechanical",     "Other Than",1),
        (FailureMode.OVERHEATS,        "temperature",    "More",      2),
    ],
    NodeType.COMPRESSOR: [
        (FailureMode.FAILS_CLOSED,     "flow",           "No/None",   3),
        (FailureMode.FAILS_HIGH,       "pressure",       "More",      3),
        (FailureMode.LOSES_POWER,      "flow",           "No/None",   2),
        (FailureMode.LEAKS,            "seal",           "As Well As",2),
        (FailureMode.VIBRATES,         "mechanical",     "Other Than",2),
        (FailureMode.REVERSE_FLOW,     "flow",           "Reverse",   2),
    ],
    NodeType.VALVE: [
        (FailureMode.FAILS_CLOSED,     "flow",           "No/None",   4),
        (FailureMode.FAILS_OPEN,       "flow",           "More",      4),
        (FailureMode.FAILS_SPURIOUS,   "flow",           "Other Than",3),
        (FailureMode.PLUGS,            "flow",           "No/None",   3),
        (FailureMode.LEAKS,            "containment",    "As Well As",2),
        (FailureMode.WRONG_SIGNAL,     "flow",           "Other Than",2),
        (FailureMode.HUMAN_ERROR,      "procedure",      "Other Than",2),
    ],
    NodeType.COLUMN: [
        (FailureMode.FAILS_HIGH,       "level",          "More",      3),
        (FailureMode.FAILS_LOW,        "level",          "No/None",   3),
        (FailureMode.LOSES_POWER,      "reflux",         "No/None",   3),
        (FailureMode.FAILS_HIGH,       "pressure",       "More",      2),
        (FailureMode.FOULING,          "trays/packing",  "Less",      2),
        (FailureMode.FAILS_CLOSED,     "reboiler",       "No/None",   2),
        (FailureMode.HUMAN_ERROR,      "procedure",      "Other Than",2),
    ],
    NodeType.SEPARATOR: [
        (FailureMode.FAILS_HIGH,       "level",          "More",      3),
        (FailureMode.FAILS_LOW,        "level",          "No/None",   3),
        (FailureMode.FAILS_OPEN,       "gas outlet",     "More",      2),
        (FailureMode.FOULING,          "internals",      "Less",      2),
    ],
    NodeType.FURNACE: [
        (FailureMode.OVERHEATS,        "temperature",    "More",      3),
        (FailureMode.LOSES_POWER,      "fuel",           "No/None",   2),
        (FailureMode.FAILS_CLOSED,     "fuel flow",      "No/None",   2),
        (FailureMode.LEAKS,            "tube",           "As Well As",2),
        (FailureMode.FAILS_HIGH,       "pressure",       "More",      2),
    ],
    NodeType.STORAGE: [
        (FailureMode.FAILS_HIGH,       "level",          "More",      3),
        (FailureMode.CORRODES,         "floor/shell",    "Less",      3),
        (FailureMode.LEAKS,            "roof seal",      "As Well As",2),
        (FailureMode.RUPTURES,         "shell",          "No/None",   1),
        (FailureMode.FAILS_SPURIOUS,   "vent",           "Other Than",2),
        (FailureMode.HUMAN_ERROR,      "filling",        "More",      2),
    ],
    NodeType.FILTER: [
        (FailureMode.PLUGS,            "flow",           "No/None",   4),
        (FailureMode.FOULING,          "flow",           "Less",      3),
        (FailureMode.FAILS_HIGH,       "pressure drop",  "More",      2),
        (FailureMode.LEAKS,            "housing",        "As Well As",1),
    ],
    NodeType.SCRUBBER: [
        (FailureMode.FAILS_LOW,        "level",          "No/None",   3),
        (FailureMode.FOULING,          "packing",        "Less",      2),
        (FailureMode.FAILS_CLOSED,     "liquid flow",    "No/None",   2),
    ],
    NodeType.INSTRUMENT: [
        (FailureMode.WRONG_SIGNAL,     "measurement",    "Other Than",3),
        (FailureMode.FAILS_HIGH,       "signal",         "More",      3),
        (FailureMode.FAILS_LOW,        "signal",         "Less",      3),
        (FailureMode.LOSES_POWER,      "signal",         "No/None",   2),
        (FailureMode.FAILS_SPURIOUS,   "trip",           "Other Than",2),
    ],
    NodeType.PIPE: [
        (FailureMode.PLUGS,            "flow",           "No/None",   3),
        (FailureMode.CORRODES,         "wall",           "Less",      3),
        (FailureMode.LEAKS,            "containment",    "No/None",   2),
        (FailureMode.RUPTURES,         "containment",    "No/None",   1),
        (FailureMode.REVERSE_FLOW,     "flow",           "Reverse",   2),
        (FailureMode.VIBRATES,         "mechanical",     "Other Than",1),
    ],
    NodeType.UTILITY: [
        (FailureMode.LOSES_POWER,      "utility",        "No/None",   3),
        (FailureMode.FAILS_LOW,        "utility flow",   "Less",      3),
        (FailureMode.FAILS_HIGH,       "utility",        "More",      2),
        (FailureMode.WRONG_SIGNAL,     "utility",        "Other Than",2),
    ],
    NodeType.OTHER: [
        (FailureMode.FAILS_CLOSED,     "flow",           "No/None",   2),
        (FailureMode.FAILS_OPEN,       "flow",           "More",      2),
        (FailureMode.LEAKS,            "containment",    "No/None",   2),
        (FailureMode.HUMAN_ERROR,      "procedure",      "Other Than",2),
    ],
}

# Default fallback
_DEFAULT_FAILURES = NODE_FAILURE_MODES[NodeType.OTHER]


# ══════════════════════════════════════════════════════════════════════════════
# Causal escalation rules
# Defines which failure modes on upstream nodes can CAUSE or WORSEN failures
# on downstream nodes.  format: (upstream_fm, downstream_fm, escalation_factor)
# escalation_factor multiplies the downstream likelihood.
# ══════════════════════════════════════════════════════════════════════════════

ESCALATION_RULES: list[tuple[FailureMode, FailureMode, float]] = [
    # Loss of cooling → reactor overheats
    (FailureMode.LOSES_POWER,   FailureMode.OVERHEATS,       2.0),
    (FailureMode.FAILS_CLOSED,  FailureMode.OVERHEATS,       2.0),
    # Pump failure → no flow downstream → level drop or temperature rise
    (FailureMode.FAILS_CLOSED,  FailureMode.FAILS_LOW,       1.8),
    (FailureMode.CAVITATION,    FailureMode.FAILS_LOW,       1.5),
    # Overpressure upstream → downstream vessel overpressure
    (FailureMode.FAILS_HIGH,    FailureMode.FAILS_HIGH,      1.5),
    # Leak upstream → flammable atmosphere → ignition downstream
    (FailureMode.LEAKS,         FailureMode.FAILS_SPURIOUS,  1.5),
    # Instrument wrong signal → valve wrong position
    (FailureMode.WRONG_SIGNAL,  FailureMode.FAILS_OPEN,      2.0),
    (FailureMode.WRONG_SIGNAL,  FailureMode.FAILS_CLOSED,    2.0),
    # Corrosion → leak → rupture
    (FailureMode.CORRODES,      FailureMode.LEAKS,           2.0),
    (FailureMode.LEAKS,         FailureMode.RUPTURES,        1.5),
    # Fouling → plugging → high pressure drop
    (FailureMode.FOULING,       FailureMode.PLUGS,           1.8),
    (FailureMode.PLUGS,         FailureMode.FAILS_HIGH,      2.0),
    # Catalyst degradation → reaction rate falls → accumulation of unreacted feed
    (FailureMode.CATALYST_DEGRADE, FailureMode.OVERHEATS,   1.5),
    # Reverse flow can carry contamination
    (FailureMode.REVERSE_FLOW,  FailureMode.WRONG_SIGNAL,   1.5),
    # Human error defeats safeguards
    (FailureMode.HUMAN_ERROR,   FailureMode.FAILS_SPURIOUS,  2.0),
    (FailureMode.HUMAN_ERROR,   FailureMode.FAILS_CLOSED,    2.0),
    # Loss of power cascades
    (FailureMode.LOSES_POWER,   FailureMode.FAILS_CLOSED,    2.0),
    (FailureMode.LOSES_POWER,   FailureMode.FAILS_SPURIOUS,  2.0),
]

# Severity boost when certain node types are involved
SEVERITY_BOOST: dict[NodeType, int] = {
    NodeType.REACTOR:   2,
    NodeType.STORAGE:   2,
    NodeType.FURNACE:   2,
    NodeType.COMPRESSOR:1,
    NodeType.COLUMN:    1,
}

# Severity boost when certain chemicals are present
CHEM_SEVERITY_BOOST = {
    "ethylene oxide": 3,
    "hydrogen":       2,
    "hydrogen sulfide":3,
    "ammonia":        2,
    "chlorine":       3,
    "phosgene":       3,
    "methane":        2,
    "propane":        1,
}

# Safeguard-defeat patterns
# If a scenario contains failure X and the node has safeguard Y,
# and X defeats Y, the risk score is boosted.
SAFEGUARD_DEFEAT_PATTERNS: list[tuple[FailureMode, str]] = [
    (FailureMode.FAILS_CLOSED,   "pressure relief"),
    (FailureMode.FAILS_CLOSED,   "PSV"),
    (FailureMode.FAILS_CLOSED,   "PRV"),
    (FailureMode.FAILS_CLOSED,   "rupture disc"),
    (FailureMode.WRONG_SIGNAL,   "alarm"),
    (FailureMode.WRONG_SIGNAL,   "interlock"),
    (FailureMode.WRONG_SIGNAL,   "ESD"),
    (FailureMode.LOSES_POWER,    "ESD"),
    (FailureMode.LOSES_POWER,    "SIS"),
    (FailureMode.HUMAN_ERROR,    "procedure"),
    (FailureMode.FAILS_SPURIOUS, "SIS"),
    (FailureMode.FAILS_SPURIOUS, "interlock"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Single-node failure event builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_failure_event(node: PIDNode, fm: FailureMode, param: str,
                          gw: str, likelihood: int) -> FailureEvent:
    chems = ", ".join(node.chemicals[:2]) if node.chemicals else "process fluid"
    desc  = (
        f"{node.tag} ({node.name}) — {fm.value}: "
        f"{gw} {param}"
        + (f" [{chems}]" if chems else "")
    )
    return FailureEvent(
        node_tag=node.tag,
        node_name=node.name,
        node_type=node.node_type.value,
        failure_mode=fm,
        parameter=param,
        guide_word=gw,
        description=desc,
        likelihood=likelihood,
    )


def enumerate_node_failures(node: PIDNode) -> list[FailureEvent]:
    """Return all credible single-failure events for one P&ID node."""
    modes = NODE_FAILURE_MODES.get(node.node_type, _DEFAULT_FAILURES)
    return [_build_failure_event(node, fm, param, gw, lk)
            for fm, param, gw, lk in modes]


# ══════════════════════════════════════════════════════════════════════════════
# Risk scoring
# ══════════════════════════════════════════════════════════════════════════════

def _score_scenario(events: list[FailureEvent], nodes: list[PIDNode]) -> tuple[int, int]:
    """
    Compute (severity, likelihood) for a compound scenario.
    Returns values 1–5.
    """
    node_map = {n.tag: n for n in nodes}

    # Base likelihood = product of individual likelihoods capped to 5
    # (multiple simultaneous failures are rarer)
    base_lk = max(1, min(5, round(
        sum(ev.likelihood for ev in events) / len(events) - (len(events) - 1) * 0.5
    )))

    # Apply escalation boosts
    for i in range(len(events) - 1):
        for rule_fm_up, rule_fm_dn, factor in ESCALATION_RULES:
            if (events[i].failure_mode == rule_fm_up and
                    events[i + 1].failure_mode == rule_fm_dn):
                base_lk = min(5, int(base_lk * factor))
                break

    # Base severity from node types involved
    base_sv = 2
    for ev in events:
        node = node_map.get(ev.node_tag)
        if node:
            base_sv += SEVERITY_BOOST.get(node.node_type, 0)
            # Chemical severity boost
            for chem in node.chemicals:
                boost = CHEM_SEVERITY_BOOST.get(chem.lower(), 0)
                base_sv += boost

    # More events in chain → higher severity (more barriers defeated)
    base_sv += len(events) - 1

    # Safeguard defeat boost
    for ev in events:
        node = node_map.get(ev.node_tag)
        if node:
            for defeat_fm, defeat_sg in SAFEGUARD_DEFEAT_PATTERNS:
                if ev.failure_mode == defeat_fm:
                    if any(defeat_sg.lower() in sg.lower() for sg in node.safeguards):
                        base_sv = min(5, base_sv + 1)

    return max(1, min(5, base_sv)), max(1, min(5, base_lk))


# ══════════════════════════════════════════════════════════════════════════════
# Causal connectivity filter
# ══════════════════════════════════════════════════════════════════════════════

def _is_causally_plausible(events: list[FailureEvent], system: PIDSystem) -> bool:
    """
    Return True if the chain of events is physically plausible.
    Rules:
      - depth-1: always plausible
      - depth-2+: at least one pair must be in the same node, adjacent nodes,
                  or connected by an escalation rule
    """
    if len(events) == 1:
        return True

    node_map = {n.tag: n for n in system.nodes}
    stream_pairs = {(s.from_tag, s.to_tag) for s in system.streams}

    for i in range(len(events) - 1):
        ev_a, ev_b = events[i], events[i + 1]

        # Same node
        if ev_a.node_tag == ev_b.node_tag:
            continue

        # Directly connected by stream
        if (ev_a.node_tag, ev_b.node_tag) in stream_pairs:
            continue
        if (ev_b.node_tag, ev_a.node_tag) in stream_pairs:
            continue

        # Shares a chemical
        node_a = node_map.get(ev_a.node_tag)
        node_b = node_map.get(ev_b.node_tag)
        if node_a and node_b:
            shared = set(c.lower() for c in node_a.chemicals) & \
                     set(c.lower() for c in node_b.chemicals)
            if shared:
                continue

        # Utility dependency (both depend on same utility node)
        if ev_a.failure_mode == FailureMode.LOSES_POWER or \
           ev_b.failure_mode == FailureMode.LOSES_POWER:
            continue

        # Escalation rule exists between their failure modes
        for rule_up, rule_dn, _ in ESCALATION_RULES:
            if ev_a.failure_mode == rule_up and ev_b.failure_mode == rule_dn:
                break
        else:
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Pre-enrichment headline generator (before LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _quick_headline(events: list[FailureEvent]) -> str:
    if len(events) == 1:
        ev = events[0]
        return f"{ev.node_tag}: {ev.failure_mode.value} — {ev.guide_word} {ev.parameter}"
    parts = [f"{ev.node_tag} {ev.failure_mode.value}" for ev in events]
    return " → ".join(parts)


def _quick_consequences(events: list[FailureEvent],
                        nodes: list[PIDNode]) -> list[str]:
    """Generate basic consequence text without LLM."""
    node_map = {n.tag: n for n in nodes}
    conseqs  = []

    for ev in events:
        node = node_map.get(ev.node_tag)
        chems = (", ".join(node.chemicals[:2]) if node and node.chemicals
                 else "process fluid")

        fm = ev.failure_mode
        if fm == FailureMode.RUPTURES:
            conseqs.append(f"Catastrophic release of {chems} — potential explosion / fire")
        elif fm == FailureMode.LEAKS:
            conseqs.append(f"Loss of containment — {chems} released to atmosphere")
        elif fm == FailureMode.OVERHEATS:
            conseqs.append(f"Thermal runaway — uncontrolled temperature rise")
        elif fm == FailureMode.FAILS_HIGH and "pressure" in ev.parameter:
            conseqs.append(f"Overpressure — vessel integrity threatened")
        elif fm == FailureMode.LOSES_POWER:
            conseqs.append(f"Loss of {ev.parameter} — process control degraded")
        elif fm == FailureMode.REVERSE_FLOW:
            conseqs.append(f"Back-contamination of {chems} into upstream system")
        elif fm == FailureMode.CATALYST_DEGRADE:
            conseqs.append("Unreacted feed accumulation — sudden exotherm on catalyst contact")
        elif fm in (FailureMode.FAILS_CLOSED, FailureMode.PLUGS):
            conseqs.append(f"No flow of {chems} — starvation / overheating downstream")

    # Compound scenarios always escalate
    if len(events) >= 2:
        conseqs.append(
            f"Compounding failure — {len(events)} simultaneous events defeat "
            f"multiple layers of protection"
        )
    if len(events) >= 3:
        conseqs.append(
            "Potential for major accident event — all safeguard layers defeated"
        )

    return list(dict.fromkeys(conseqs))[:5]


def _identify_safeguard_gaps(events: list[FailureEvent],
                              nodes: list[PIDNode]) -> list[str]:
    """Identify which safeguards are defeated by this scenario."""
    node_map = {n.tag: n for n in nodes}
    gaps = []
    for ev in events:
        node = node_map.get(ev.node_tag)
        if not node:
            continue
        for defeat_fm, defeat_sg in SAFEGUARD_DEFEAT_PATTERNS:
            if ev.failure_mode == defeat_fm:
                for sg in node.safeguards:
                    if defeat_sg.lower() in sg.lower():
                        gaps.append(
                            f"{ev.node_tag}: {ev.failure_mode.value} defeats "
                            f"safeguard '{sg}'"
                        )
    return gaps[:4]


# ══════════════════════════════════════════════════════════════════════════════
# Main Generator
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioGenerator:
    """
    Generates compounding failure scenarios from a PIDSystem.

    Usage:
        gen = ScenarioGenerator(max_depth=3, max_scenarios=2000)
        scenario_set = gen.run(pid_system)
    """

    def __init__(
        self,
        max_depth:      int = 3,
        max_scenarios:  int = 2000,
        min_risk_score: int = 2,
        include_single: bool = True,
        include_double: bool = True,
        include_triple: bool = True,
    ):
        self.max_depth      = max_depth
        self.max_scenarios  = max_scenarios
        self.min_risk_score = min_risk_score
        self.include_single = include_single
        self.include_double = include_double
        self.include_triple = include_triple

    # ── Step 1: enumerate all single failures ─────────────────────────────────

    def _all_failures(self, system: PIDSystem) -> list[FailureEvent]:
        events = []
        for node in system.nodes:
            events.extend(enumerate_node_failures(node))
        logger.info(f"Enumerated {len(events)} single failure events across "
                    f"{len(system.nodes)} nodes")
        return events

    # ── Step 2: build compound chains ─────────────────────────────────────────

    def _generate_chains(
        self, all_events: list[FailureEvent], system: PIDSystem
    ) -> list[list[FailureEvent]]:
        chains: list[list[FailureEvent]] = []

        if self.include_single:
            chains += [[ev] for ev in all_events]

        if self.include_double:
            # Pairs — only consider events from different nodes to avoid
            # duplicating same-node pairs
            for ev_a, ev_b in itertools.combinations(all_events, 2):
                chain = [ev_a, ev_b]
                if _is_causally_plausible(chain, system):
                    chains.append(chain)

        if self.include_triple and self.max_depth >= 3:
            # Triples — only between distinct nodes, causal filter applied
            unique_events_by_node: dict[str, list[FailureEvent]] = {}
            for ev in all_events:
                unique_events_by_node.setdefault(ev.node_tag, []).append(ev)

            # Pick one representative event per node to keep combinations tractable
            node_representatives = [
                evs[0] for evs in unique_events_by_node.values()
            ]

            for trio in itertools.combinations(node_representatives, 3):
                chain = list(trio)
                if _is_causally_plausible(chain, system):
                    chains.append(chain)

        logger.info(f"Generated {len(chains)} raw chains "
                    f"(single={self.include_single}, double={self.include_double}, "
                    f"triple={self.include_triple})")
        return chains

    # ── Step 3: build CompoundScenarios from chains ───────────────────────────

    def _build_scenarios(
        self, chains: list[list[FailureEvent]], system: PIDSystem
    ) -> list[CompoundScenario]:
        scenarios = []
        nodes = system.nodes

        for chain in chains:
            sev, lk = _score_scenario(chain, nodes)
            risk_score = sev * lk

            if risk_score < self.min_risk_score:
                continue

            sc = CompoundScenario(
                events=chain,
                headline=_quick_headline(chain),
                consequences=_quick_consequences(chain, nodes),
                safeguard_gaps=_identify_safeguard_gaps(chain, nodes),
                severity=sev,
                likelihood=lk,
                risk_score=risk_score,
                compound_depth=len(chain),
            )
            sc.recompute_risk()
            scenarios.append(sc)

        # Sort by risk_score descending, then depth descending
        scenarios.sort(key=lambda s: (-s.risk_score, -s.compound_depth))

        # Cap at max_scenarios
        if len(scenarios) > self.max_scenarios:
            scenarios = scenarios[:self.max_scenarios]

        logger.info(
            f"Built {len(scenarios)} scenarios "
            f"(capped at {self.max_scenarios}, min_risk={self.min_risk_score})"
        )
        return scenarios

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, system: PIDSystem) -> ScenarioSet:
        """
        Full generation pipeline for one PIDSystem.
        Returns a ScenarioSet ready for enrichment and ranking.
        """
        from datetime import datetime

        logger.info(f"Starting scenario generation for: {system.name}")

        all_events = self._all_failures(system)
        chains     = self._generate_chains(all_events, system)
        scenarios  = self._build_scenarios(chains, system)

        ss = ScenarioSet(
            system_name=system.name,
            pid_system=system.to_dict(),
            scenarios=scenarios,
            generated_at=datetime.utcnow().isoformat(),
        )
        ss.compute_stats()

        logger.info(
            f"Scenario generation complete: {ss.total} scenarios | "
            + " | ".join(f"{k}={v}" for k, v in ss.by_risk.items())
        )
        return ss
