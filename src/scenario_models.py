"""
scenario_models.py — Core data models for HAZOP Scenario Generation.

Hierarchy:
  PIDNode          — one process node from the P&ID (vessel, pump, HX, etc.)
  PIDStream        — a connecting stream between two nodes
  PIDSystem        — the complete parsed P&ID (nodes + streams + instruments)
  FailureEvent     — a single equipment/instrument/utility failure
  CompoundScenario — a chain of 1–N FailureEvents with consequences
  ScenarioSet      — the full set of generated scenarios for one P&ID system
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import uuid
import json


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class NodeType(str, Enum):
    REACTOR         = "Reactor"
    VESSEL          = "Vessel / Tank"
    COLUMN          = "Distillation Column"
    HEAT_EXCHANGER  = "Heat Exchanger"
    PUMP            = "Pump"
    COMPRESSOR      = "Compressor"
    VALVE           = "Valve"
    PIPE            = "Pipe / Line"
    INSTRUMENT      = "Instrument"
    SEPARATOR       = "Separator"
    FURNACE         = "Furnace / Fired Heater"
    STORAGE         = "Storage Tank"
    SCRUBBER        = "Scrubber / Absorber"
    FILTER          = "Filter / Strainer"
    UTILITY         = "Utility System"
    OTHER           = "Other"


class FailureMode(str, Enum):
    FAILS_CLOSED     = "Fails Closed"
    FAILS_OPEN       = "Fails Open"
    FAILS_HIGH       = "Fails High"
    FAILS_LOW        = "Fails Low"
    FAILS_SPURIOUS   = "Spurious Trip / Activation"
    LEAKS            = "External Leak"
    RUPTURES         = "Catastrophic Rupture"
    PLUGS            = "Plugs / Blocks"
    CORRODES         = "Corrodes / Degrades"
    OVERHEATS        = "Overheats"
    LOSES_POWER      = "Loses Power / Utility"
    WRONG_SIGNAL     = "Wrong / Erroneous Signal"
    REVERSE_FLOW     = "Reverse Flow"
    VIBRATES         = "Excessive Vibration"
    HUMAN_ERROR      = "Human Error / Procedure Deviation"
    CATALYST_DEGRADE = "Catalyst Degradation"
    FOULING          = "Fouling / Scale Buildup"
    CAVITATION       = "Cavitation"


class RiskLevel(str, Enum):
    CRITICAL = "Critical"
    HIGH     = "High"
    MEDIUM   = "Medium"
    LOW      = "Low"


class ScenarioStatus(str, Enum):
    NEW         = "New"
    REVIEWED    = "Reviewed"
    ACCEPTED    = "Accepted"
    REJECTED    = "Rejected / Not Credible"
    ACTION_OPEN = "Action Open"


# ══════════════════════════════════════════════════════════════════════════════
# P&ID Data Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DesignCondition:
    """Normal operating / design limits for a process node."""
    temperature_c:  Optional[float] = None   # °C
    pressure_barg:  Optional[float] = None   # barg
    flow_kgh:       Optional[float] = None   # kg/h
    level_pct:      Optional[float] = None   # %
    phase:          str = "liquid"           # liquid | gas | mixed
    moc:            str = ""                 # material of construction
    design_temp_c:  Optional[float] = None
    design_press_barg: Optional[float] = None
    mawp_barg:      Optional[float] = None   # Max Allowable Working Pressure


@dataclass
class Instrument:
    tag:         str              # e.g. FIC-101
    type:        str              # flow / pressure / temperature / level / analyser
    function:    str              # indication / control / alarm / trip
    set_point:   Optional[float] = None
    action:      str = ""        # e.g. "close feed valve on high-high"
    is_sis:      bool = False     # part of Safety Instrumented System


@dataclass
class PIDNode:
    """One equipment item on the P&ID."""
    tag:           str                        # e.g. V-101, R-201
    name:          str                        # e.g. "Feed Surge Drum"
    node_type:     NodeType = NodeType.VESSEL
    chemicals:     list[str] = field(default_factory=list)
    conditions:    DesignCondition = field(default_factory=DesignCondition)
    instruments:   list[Instrument] = field(default_factory=list)
    safeguards:    list[str] = field(default_factory=list)
    description:   str = ""
    upstream_tags: list[str] = field(default_factory=list)
    downstream_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def short_label(self) -> str:
        return f"{self.tag} ({self.name})"


@dataclass
class PIDStream:
    """A process stream connecting two nodes."""
    stream_id:   str
    from_tag:    str
    to_tag:      str
    chemicals:   list[str] = field(default_factory=list)
    phase:       str = "liquid"
    line_size_mm: Optional[float] = None
    has_check_valve: bool = False
    has_isolation:   bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PIDSystem:
    """Complete parsed representation of a P&ID."""
    system_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str = "Process System"
    description: str = ""
    nodes:       list[PIDNode]   = field(default_factory=list)
    streams:     list[PIDStream] = field(default_factory=list)
    utilities:   list[str]       = field(default_factory=list)   # CW, steam, N2…
    raw_text:    str = ""

    def node_by_tag(self, tag: str) -> Optional[PIDNode]:
        for n in self.nodes:
            if n.tag.upper() == tag.upper():
                return n
        return None

    def downstream_nodes(self, tag: str) -> list[PIDNode]:
        result = []
        for s in self.streams:
            if s.from_tag.upper() == tag.upper():
                n = self.node_by_tag(s.to_tag)
                if n:
                    result.append(n)
        return result

    def upstream_nodes(self, tag: str) -> list[PIDNode]:
        result = []
        for s in self.streams:
            if s.to_tag.upper() == tag.upper():
                n = self.node_by_tag(s.from_tag)
                if n:
                    result.append(n)
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"System: {self.name} | "
            f"Nodes: {len(self.nodes)} | "
            f"Streams: {len(self.streams)} | "
            f"Utilities: {', '.join(self.utilities)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Failure & Scenario Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureEvent:
    """A single failure on one node — the atomic unit of a scenario chain."""
    event_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_tag:     str = ""
    node_name:    str = ""
    node_type:    str = ""
    failure_mode: FailureMode = FailureMode.FAILS_CLOSED
    parameter:    str = ""      # flow / pressure / temperature / level / concentration
    guide_word:   str = ""      # No | More | Less | Reverse | Other Than | As Well As
    description:  str = ""      # "Cooling water control valve FCV-101 fails closed"
    likelihood:   int = 2       # 1–5

    def label(self) -> str:
        return f"{self.node_tag}: {self.failure_mode.value} → {self.guide_word} {self.parameter}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_mode"] = self.failure_mode.value
        return d


@dataclass
class CompoundScenario:
    """
    A chain of 1–N FailureEvents that interact to produce a hazardous outcome.
    This is the core output of the generator.

    Example 3-event chain:
      1. Cooling water valve FCV-101 fails closed   (No flow)
      2. Catalyst activity rises with temperature   (More reaction rate)
      3. Pressure relief valve PSV-201 fails to open (No safeguard)
      → Consequence: Thermal runaway → reactor overpressure → catastrophic rupture
    """
    scenario_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    events:        list[FailureEvent] = field(default_factory=list)

    # Populated by GenAI enricher
    headline:      str = ""     # one-line summary
    mechanism:     str = ""     # narrative causal chain explanation
    consequences:  list[str] = field(default_factory=list)
    top_consequence: str = ""   # worst-case single consequence
    safeguard_gaps: list[str] = field(default_factory=list)
    existing_safeguards: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    historical_precedent: str = ""  # real incident this resembles

    # Risk
    severity:      int = 3      # 1–5
    likelihood:    int = 2      # 1–5
    risk_score:    int = 0      # severity × likelihood (computed)
    risk_level:    RiskLevel = RiskLevel.MEDIUM
    action_priority: str = "Short-term"

    # Metadata
    compound_depth: int = 1     # number of simultaneous failures
    status:         ScenarioStatus = ScenarioStatus.NEW
    tags:           list[str] = field(default_factory=list)
    llm_enriched:   bool = False
    reviewer_notes: str = ""

    def __post_init__(self):
        self.compound_depth = len(self.events)
        if self.risk_score == 0:
            self.risk_score = self.severity * self.likelihood
        self._set_risk_level()

    def _set_risk_level(self):
        s = self.risk_score
        if s >= 15:
            self.risk_level    = RiskLevel.CRITICAL
            self.action_priority = "Immediate"
        elif s >= 9:
            self.risk_level    = RiskLevel.HIGH
            self.action_priority = "Immediate"
        elif s >= 4:
            self.risk_level    = RiskLevel.MEDIUM
            self.action_priority = "Short-term"
        else:
            self.risk_level    = RiskLevel.LOW
            self.action_priority = "Long-term"

    def recompute_risk(self):
        self.risk_score = self.severity * self.likelihood
        self._set_risk_level()

    def event_chain_text(self) -> str:
        """Plain-text ordered chain for display."""
        lines = []
        for i, ev in enumerate(self.events, 1):
            lines.append(f"  {i}. [{ev.node_tag}] {ev.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_level"]   = self.risk_level.value
        d["status"]       = self.status.value
        for ev in d["events"]:
            ev["failure_mode"] = ev.get("failure_mode", "")
        return d

    def short_id(self) -> str:
        return f"SC-{self.scenario_id.upper()[:6]}"


@dataclass
class ScenarioSet:
    """All generated scenarios for one P&ID system."""
    set_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    system_name:  str = "Process System"
    pid_system:   Optional[dict] = None       # serialised PIDSystem
    scenarios:    list[CompoundScenario] = field(default_factory=list)
    generated_at: str = ""
    generator_version: str = "1.0"

    # Statistics (computed)
    total:        int = 0
    by_risk:      dict = field(default_factory=dict)
    by_depth:     dict = field(default_factory=dict)
    by_node:      dict = field(default_factory=dict)

    def compute_stats(self):
        self.total = len(self.scenarios)
        self.by_risk  = {lvl.value: 0 for lvl in RiskLevel}
        self.by_depth = {}
        self.by_node  = {}
        for sc in self.scenarios:
            self.by_risk[sc.risk_level.value] += 1
            d = str(sc.compound_depth)
            self.by_depth[d] = self.by_depth.get(d, 0) + 1
            for ev in sc.events:
                tag = ev.node_tag
                self.by_node[tag] = self.by_node.get(tag, 0) + 1

    def top_scenarios(self, n: int = 20) -> list[CompoundScenario]:
        return sorted(self.scenarios, key=lambda s: (-s.risk_score, -s.compound_depth))[:n]

    def filter(
        self,
        min_risk_score: int = 0,
        risk_levels: Optional[list[str]] = None,
        min_depth: int = 1,
        node_tags: Optional[list[str]] = None,
    ) -> list[CompoundScenario]:
        result = []
        for sc in self.scenarios:
            if sc.risk_score < min_risk_score:
                continue
            if risk_levels and sc.risk_level.value not in risk_levels:
                continue
            if sc.compound_depth < min_depth:
                continue
            if node_tags:
                sc_tags = {ev.node_tag for ev in sc.events}
                if not sc_tags.intersection(node_tags):
                    continue
            result.append(sc)
        return sorted(result, key=lambda s: -s.risk_score)

    def to_dict(self) -> dict:
        self.compute_stats()
        return {
            "set_id":            self.set_id,
            "system_name":       self.system_name,
            "generated_at":      self.generated_at,
            "generator_version": self.generator_version,
            "stats": {
                "total":    self.total,
                "by_risk":  self.by_risk,
                "by_depth": self.by_depth,
                "by_node":  self.by_node,
            },
            "scenarios": [sc.to_dict() for sc in self.scenarios],
        }

    def save(self, path) -> None:
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ScenarioSet":
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ss = cls(
            set_id=data.get("set_id", ""),
            system_name=data.get("system_name", ""),
            generated_at=data.get("generated_at", ""),
        )
        for sc_dict in data.get("scenarios", []):
            events = [
                FailureEvent(**{k: v for k, v in ev.items() if k != "failure_mode"})
                for ev in sc_dict.get("events", [])
            ]
            sc = CompoundScenario(
                scenario_id=sc_dict.get("scenario_id", ""),
                events=events,
                headline=sc_dict.get("headline", ""),
                mechanism=sc_dict.get("mechanism", ""),
                consequences=sc_dict.get("consequences", []),
                top_consequence=sc_dict.get("top_consequence", ""),
                safeguard_gaps=sc_dict.get("safeguard_gaps", []),
                existing_safeguards=sc_dict.get("existing_safeguards", []),
                recommendations=sc_dict.get("recommendations", []),
                historical_precedent=sc_dict.get("historical_precedent", ""),
                severity=sc_dict.get("severity", 3),
                likelihood=sc_dict.get("likelihood", 2),
                risk_score=sc_dict.get("risk_score", 0),
                compound_depth=sc_dict.get("compound_depth", 1),
                llm_enriched=sc_dict.get("llm_enriched", False),
                reviewer_notes=sc_dict.get("reviewer_notes", ""),
            )
            sc.recompute_risk()
            ss.scenarios.append(sc)
        ss.compute_stats()
        return ss
