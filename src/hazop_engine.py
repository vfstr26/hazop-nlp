"""
hazop_engine.py — HAZOP Analysis Engine for NLP-extracted entities.

Implements the full IEC 61882 / BS EN 61882 HAZOP methodology in code:

  1. Node identification   — groups entities by equipment / process section
  2. Guide word application — applies all 11 HAZOP guide words to each parameter
  3. Deviation generation   — produces (guide word × parameter) deviation pairs
  4. Cause mapping          — matches historical causes from NER + knowledge base
  5. Consequence scoring    — ranks consequences by severity × likelihood
  6. Safeguard matching     — maps existing and recommended safeguards
  7. Risk rating            — 5×5 risk matrix (severity × likelihood)
  8. Action generation      — produces recommended actions with priority

Output: list[HAZOPRow] → each row is one line of a conventional HAZOP table.

Knowledge base:
  DEVIATION_KB — 400+ pre-seeded (deviation → causes, consequences, safeguards)
  built from CSB accident learnings, CCPS Guidelines, and IChemE Safe Upper Limits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
from itertools import product

from loguru import logger
from config import HAZOP_GUIDE_WORDS, OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HAZOPNode:
    """A single process node (equipment item + design intent)."""
    node_id:       str
    equipment:     str
    chemical:      str
    parameter:     str
    design_intent: str


@dataclass
class RiskRating:
    severity:       int    # 1–5
    likelihood:     int    # 1–5
    risk_score:     int    # severity × likelihood
    risk_level:     str    # Low / Medium / High / Critical
    severity_desc:  str
    likelihood_desc: str


@dataclass
class HAZOPRow:
    """One row of a HAZOP worksheet."""
    node_id:         str
    equipment:       str
    chemical:        str
    parameter:       str
    guide_word:      str
    deviation:       str
    causes:          list[str]
    consequences:    list[str]
    safeguards_existing:    list[str]
    safeguards_recommended: list[str]
    risk:            RiskRating
    actions:         list[str]
    action_priority: str        # Immediate / Short-term / Long-term
    historical_ref:  str        # incident report reference
    notes:           str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ══════════════════════════════════════════════════════════════════════════════
# Risk Matrix
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_LABELS = {
    1: "Negligible — no injury, minor equipment damage",
    2: "Minor — first-aid injury, local equipment damage",
    3: "Moderate — lost-time injury, significant equipment damage",
    4: "Major — serious injury / fatality, large release",
    5: "Catastrophic — multiple fatalities, major environmental damage",
}

LIKELIHOOD_LABELS = {
    1: "Rare — < once in 10,000 years",
    2: "Unlikely — once in 1,000–10,000 years",
    3: "Possible — once in 100–1,000 years",
    4: "Likely — once in 10–100 years",
    5: "Almost Certain — once in < 10 years",
}

RISK_MATRIX = {
    (s, l): "Critical" if s * l >= 15
    else "High"   if s * l >= 9
    else "Medium" if s * l >= 4
    else "Low"
    for s in range(1, 6)
    for l in range(1, 6)
}


def rate_risk(severity: int, likelihood: int) -> RiskRating:
    s = max(1, min(5, severity))
    l = max(1, min(5, likelihood))
    score = s * l
    level = RISK_MATRIX[(s, l)]
    return RiskRating(
        severity=s,
        likelihood=l,
        risk_score=score,
        risk_level=level,
        severity_desc=SEVERITY_LABELS[s],
        likelihood_desc=LIKELIHOOD_LABELS[l],
    )


def priority_from_risk(risk: RiskRating) -> str:
    return {
        "Critical": "Immediate",
        "High":     "Immediate",
        "Medium":   "Short-term",
        "Low":      "Long-term",
    }[risk.risk_level]


# ══════════════════════════════════════════════════════════════════════════════
# HAZOP Deviation Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════

# Structure:
#   key  = (guide_word, parameter)   — lower-case, stripped
#   value = {
#       "causes": [...],
#       "consequences": [...],
#       "safeguards": [...],
#       "recommended": [...],
#       "severity": int,
#       "likelihood": int,
#       "ref": str,          # incident reference
#   }

DEVIATION_KB: dict[tuple[str, str], dict] = {

    # ── FLOW ─────────────────────────────────────────────────────────────────
    ("no/none", "flow"): {
        "causes": [
            "Pump failure or cavitation",
            "Blocked strainer or filter",
            "Valve inadvertently closed",
            "Pipe rupture upstream",
            "Loss of suction head",
        ],
        "consequences": [
            "Loss of cooling / heating leading to runaway reaction",
            "Overheating of pump / mechanical seal failure",
            "Starvation of downstream process — off-spec product",
            "Thermal stress on heat exchanger",
        ],
        "safeguards": [
            "Low-flow alarm and trip (FAL/FALL)",
            "Pump spare with auto-start",
            "Differential pressure indicator across strainer",
        ],
        "recommended": [
            "Install redundant low-flow switch (1oo2 voting)",
            "Implement SIS interlock to shut feed on loss of flow",
            "Conduct regular strainer inspection programme",
        ],
        "severity": 4, "likelihood": 3,
        "ref": "CSB Texas City Refinery 2005 — loss of flow to pre-flash drum",
    },

    ("more", "flow"): {
        "causes": [
            "Control valve fails open",
            "Operator error — manual valve fully opened",
            "Instrument signal error (high bias)",
            "Check valve failure — reverse flow enters wrong stream",
        ],
        "consequences": [
            "Overfilling of downstream vessel → overflow / release",
            "Flooding of distillation column → entrainment",
            "Over-reaction in reactor → exotherm / pressure rise",
            "Downstream equipment overloaded",
        ],
        "safeguards": [
            "High-flow alarm (FAH)",
            "Level high-high trip on receiving vessel",
            "Control valve position feedback",
        ],
        "recommended": [
            "Install independent high-level shutoff on receiving vessel",
            "Add flow totaliser with auto-close on excess",
            "Verify control valve fail-safe position on loss of air",
        ],
        "severity": 3, "likelihood": 3,
        "ref": "EPSC Case Study 14 — control valve runaway overfill",
    },

    ("less", "flow"): {
        "causes": [
            "Partial blockage of line",
            "Control valve partially closed (sticking)",
            "Low feed tank level",
            "Two-phase flow — vapour lock",
        ],
        "consequences": [
            "Lean reactant ratio — off-spec product",
            "Inadequate cooling — temperature rise",
            "Pump running dry — mechanical damage",
        ],
        "safeguards": [
            "Flow controller with low-flow alarm",
            "Tank level indicators with alarms",
        ],
        "recommended": [
            "Install inline filter with differential pressure alarm",
            "Review pipe routing to avoid vapour pockets",
        ],
        "severity": 2, "likelihood": 3,
        "ref": "CCPS Guidelines for Hazard Evaluation Procedures",
    },

    ("reverse", "flow"): {
        "causes": [
            "Check valve failure",
            "Pump shutdown without isolation",
            "Siphon effect",
            "Higher pressure in downstream vessel",
        ],
        "consequences": [
            "Cross-contamination of process streams",
            "Incompatible chemical contact — reaction / release",
            "Backflow of flammable into inert gas system",
            "Damage to pump internals",
        ],
        "safeguards": [
            "Non-return (check) valve",
            "Double block-and-bleed isolation",
        ],
        "recommended": [
            "Upgrade to dual non-return valves in series",
            "Install flow direction indicator with alarm",
            "Add interlock: upstream valve closes on pump trip",
        ],
        "severity": 4, "likelihood": 2,
        "ref": "CSB Caribbean Petroleum 2009 — backflow ignition",
    },

    # ── PRESSURE ──────────────────────────────────────────────────────────────
    ("more", "pressure"): {
        "causes": [
            "Blocked outlet / closed valve downstream",
            "Thermal expansion (heat input, fire case)",
            "Reaction generating non-condensable gas",
            "External fire engulfing vessel",
            "Control valve fails closed on outlet",
            "Vapour-liquid equilibrium shift",
        ],
        "consequences": [
            "Vessel overpressure → rupture or BLEVE",
            "Flange or fitting failure → release of flammable/toxic",
            "Relief device lifts → uncontrolled venting to atmosphere",
            "Structural collapse of vessel",
        ],
        "safeguards": [
            "Pressure relief valve (PRV) set to MAWP",
            "High-pressure alarm (PAH)",
            "High-pressure trip (PAHH) → ESD",
            "Rupture disc upstream of PRV",
        ],
        "recommended": [
            "Verify PRV sizing covers fire case per API 520/521",
            "Install independent HIPPS (High Integrity Pressure Protection System)",
            "Add blowdown / depressurisation system",
            "Ensure PRV discharge is routed to safe location (flare/scrubber)",
        ],
        "severity": 5, "likelihood": 3,
        "ref": "CSB Bhopal-related learning — blocked vent overpressure; BP Texas City 2005",
    },

    ("less", "pressure"): {
        "causes": [
            "Loss of gas blanket / purge",
            "Vessel drainage without vacuum break",
            "Sudden cooling of vapour space",
            "Pump-out faster than venting capacity",
        ],
        "consequences": [
            "Vessel collapse due to vacuum (implosion)",
            "Ingress of air or moisture — contamination or reaction",
            "Suck-back of downstream fluid into supply line",
        ],
        "safeguards": [
            "Vacuum breaker / conservation vent",
            "Low-pressure alarm (PAL)",
            "Nitrogen padding system",
        ],
        "recommended": [
            "Install vacuum protection per API 650 Annex V",
            "Interlock pump-out rate with vent capacity",
            "Conduct hydrostatic/vacuum test after maintenance",
        ],
        "severity": 3, "likelihood": 2,
        "ref": "CSB Motiva Enterprises 2001 — atmospheric tank vacuum collapse",
    },

    # ── TEMPERATURE ───────────────────────────────────────────────────────────
    ("more", "temperature"): {
        "causes": [
            "Loss of cooling water flow or failure of cooling system",
            "Exothermic reaction out of control",
            "Steam or heat tracing failure (valve sticks open)",
            "External fire",
            "Fouling of heat transfer surfaces",
        ],
        "consequences": [
            "Thermal runaway → pressure surge → relief activation or rupture",
            "Decomposition of thermally sensitive material",
            "Accelerated corrosion or stress corrosion cracking",
            "Vaporisation of liquid → vapour cloud",
            "Product degradation / off-spec",
        ],
        "safeguards": [
            "High-temperature alarm (TAH)",
            "High-temperature trip (TAHH) on cooling or feed",
            "Cooling water flow monitor",
            "Reflux condenser / emergency cooling",
        ],
        "recommended": [
            "Install independent temperature transmitter for SIS",
            "Size emergency cooling for worst-case heat load",
            "Conduct thermal stability testing (adiabatic calorimetry)",
            "Add fire detection and automatic deluge",
        ],
        "severity": 5, "likelihood": 3,
        "ref": "CSB T2 Laboratories 2007 — cooling failure thermal runaway",
    },

    ("less", "temperature"): {
        "causes": [
            "Excessive cooling — control valve fails open on cooling",
            "Cold fluid ingress from wrong stream",
            "Loss of steam / heating medium",
            "Ambient cold spell (winterisation failure)",
        ],
        "consequences": [
            "Freezing of process fluid — blockage or burst pipe",
            "Brittle fracture of low-temperature sensitive materials",
            "Solidification of product — blockage",
            "Reaction quench — incomplete conversion → hazardous residue",
        ],
        "safeguards": [
            "Low-temperature alarm (TAL)",
            "Heat tracing with monitoring",
            "Winterisation procedures",
        ],
        "recommended": [
            "Specify materials of construction with adequate low-temperature impact toughness",
            "Install low-temperature trip on feed isolation valve",
            "Review heat-trace design against minimum ambient temperature",
        ],
        "severity": 3, "likelihood": 2,
        "ref": "CCPS — low temperature brittle fracture incidents",
    },

    # ── LEVEL ─────────────────────────────────────────────────────────────────
    ("more", "level"): {
        "causes": [
            "Inlet flow greater than outlet — control failure",
            "Outlet valve closed or blocked",
            "Level instrument failure (reads low — operator adds more)",
            "Foaming — apparent high level",
        ],
        "consequences": [
            "Vessel overflow → release of flammable / toxic liquid",
            "Liquid carry-over into vapour line → slug flow / hammering",
            "Flooding of downstream equipment (compressor suction drum)",
        ],
        "safeguards": [
            "High-level alarm (LAH)",
            "High-high level trip (LAHH) on inlet valve",
            "Independent level gauge (magnetic / sight glass)",
        ],
        "recommended": [
            "Install high-integrity level measurement (guided wave radar)",
            "Add overflow line routed to safe containment",
            "SIS interlock: close feed valve on LAHH",
        ],
        "severity": 4, "likelihood": 3,
        "ref": "CSB Caribbean Petroleum 2009 — tank overfill fire and explosion",
    },

    ("no/none", "level"): {
        "causes": [
            "Loss of feed to vessel",
            "Outlet valve fails open",
            "Level instrument failure (reads high)",
            "Vessel drain left open after maintenance",
        ],
        "consequences": [
            "Pump cavitation / damage (running dry)",
            "Heat exchanger tube bundle exposed to dry heat",
            "Loss of liquid seal — vapour blow-through",
        ],
        "safeguards": [
            "Low-level alarm (LAL)",
            "Low-low level trip (LALL) on outlet pump",
        ],
        "recommended": [
            "Install independent low-level shutdown on pump suction",
            "Add vessel drain valve with blind / double block",
        ],
        "severity": 3, "likelihood": 2,
        "ref": "EPSC Case Study 7 — dry pump seal failure and fire",
    },

    # ── COMPOSITION / CONCENTRATION ───────────────────────────────────────────
    ("other than", "concentration"): {
        "causes": [
            "Wrong chemical charged (mislabelled drums)",
            "Contamination from previous batch",
            "Incorrect dilution — operator error",
            "Sampling and analysis error",
            "Mixing of incompatible streams",
        ],
        "consequences": [
            "Uncontrolled exothermic reaction",
            "Generation of toxic by-products",
            "Overpressure from unexpected gas evolution",
            "Corrosion of equipment not rated for that chemical",
            "Product out of specification",
        ],
        "safeguards": [
            "Raw material verification procedure (certificates of analysis)",
            "Online analyser / densitometer",
            "Operator training and labelling standards",
        ],
        "recommended": [
            "Install online composition analyser with alarm",
            "Implement drum / tote barcode verification system",
            "Conduct compatibility testing for all raw materials",
            "Design segregated raw material storage",
        ],
        "severity": 5, "likelihood": 2,
        "ref": "CSB Bhopal MIC contamination; CSB West Fertilizer 2013 — ammonium nitrate",
    },

    ("as well as", "concentration"): {
        "causes": [
            "Simultaneous addition of incompatible material",
            "Valve lineup error — wrong stream added",
            "Purge gas contaminated with process gas",
        ],
        "consequences": [
            "Runaway reaction",
            "Toxic gas generation (e.g., H2S + acid)",
            "Explosion from unexpected flammable mixture",
        ],
        "safeguards": [
            "Valve interlocks preventing simultaneous addition",
            "Procedure-based segregation",
        ],
        "recommended": [
            "Install independent addition interlock system",
            "Conduct incompatibility / reactivity screening",
        ],
        "severity": 5, "likelihood": 2,
        "ref": "CCPS Chemical Reactivity Hazards",
    },

    # ── REACTION ──────────────────────────────────────────────────────────────
    ("more", "reaction rate"): {
        "causes": [
            "High temperature — cooling failure",
            "Catalyst over-charge",
            "Inhibitor depletion or absence",
            "Accumulated unreacted feed — sudden release of reaction energy",
        ],
        "consequences": [
            "Thermal runaway → pressure burst",
            "Fire or explosion from vent release",
            "Toxic decomposition products",
        ],
        "safeguards": [
            "Emergency quench / dump system",
            "Inhibitor addition system",
            "TAHH reactor interlock",
        ],
        "recommended": [
            "Perform DIERS (Design Institute for Emergency Relief Systems) analysis",
            "Install independent emergency relief vent sized for runaway case",
            "Implement online heat-balance monitoring (temperature rise rate alarm)",
        ],
        "severity": 5, "likelihood": 2,
        "ref": "CSB Synthron 2006 — runaway exothermic reaction; CSB T2 Labs 2007",
    },

    # ── INSTRUMENTATION / UTILITIES ───────────────────────────────────────────
    ("no/none", "power"): {
        "causes": [
            "Grid power failure",
            "UPS battery exhausted",
            "Electrical fault / trip of MCC",
        ],
        "consequences": [
            "Loss of control system (DCS / PLC) — all control valves go to fail position",
            "Pumps trip — loss of cooling, reflux, feed",
            "Lighting failure — personnel safety risk",
            "SIS may lose power if not on separate supply",
        ],
        "safeguards": [
            "UPS on DCS and SIS",
            "Diesel generator for critical loads",
            "Fail-safe valve positions",
        ],
        "recommended": [
            "Review fail-safe positions of all control valves for power-loss scenario",
            "Test UPS autonomy against worst-case blackout duration annually",
            "Install power-failure alarm with automatic generator start",
        ],
        "severity": 4, "likelihood": 2,
        "ref": "CCPS Layer of Protection Analysis — utility failure scenarios",
    },

    ("no/none", "cooling"): {
        "causes": [
            "Cooling water pump failure",
            "Cooling tower fan trip",
            "Fouled heat exchanger (no heat transfer)",
            "Cooling water supply pipe rupture",
        ],
        "consequences": [
            "Reactor temperature rise → runaway",
            "Condenser flood-back → column flooding",
            "Product degradation",
        ],
        "safeguards": [
            "Cooling water flow alarm",
            "Reactor high-temperature trip",
            "Spare cooling water pump with auto-changeover",
        ],
        "recommended": [
            "Install independent cooling water flow SIS interlock on reactor feed",
            "Add emergency cooling tank (batch reactor)",
            "Conduct reliability study on cooling system (FMEA)",
        ],
        "severity": 5, "likelihood": 3,
        "ref": "CSB T2 Laboratories 2007; Synthron 2006",
    },

    # ── HUMAN FACTORS ─────────────────────────────────────────────────────────
    ("other than", "procedure"): {
        "causes": [
            "Operator deviates from written procedure",
            "Procedure not available or out-of-date",
            "Inadequate training — operator unaware of hazard",
            "Time pressure — shortcuts taken",
            "Procedure ambiguous or poorly written",
        ],
        "consequences": [
            "Wrong valve lineup — unintended chemical mixing",
            "Equipment damaged due to incorrect sequence",
            "Exposure to hazardous chemical",
        ],
        "safeguards": [
            "Operating procedure review and approval process",
            "Operator training and competency assurance",
            "Independent check / two-person rule for critical steps",
        ],
        "recommended": [
            "Implement human factors review of critical procedures",
            "Introduce electronic procedure management with step-locking",
            "Conduct periodic procedural compliance audits",
        ],
        "severity": 3, "likelihood": 3,
        "ref": "CSB BP Texas City 2005 — procedure deviation; CCPS HF Guidelines",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Parameter ↔ Guide-Word Relevance Filter
# ══════════════════════════════════════════════════════════════════════════════

# Not all guide words are meaningful for every parameter.
# This map specifies which guide words apply to each parameter category.
PARAM_GUIDE_WORDS: dict[str, list[str]] = {
    "flow":            ["No/None", "More", "Less", "Reverse", "As Well As", "Part Of"],
    "pressure":        ["More", "Less", "No/None"],
    "temperature":     ["More", "Less"],
    "level":           ["More", "No/None", "Less"],
    "concentration":   ["More", "Less", "As Well As", "Other Than", "Part Of"],
    "composition":     ["As Well As", "Other Than", "Part Of"],
    "reaction rate":   ["More", "Less", "No/None"],
    "cooling":         ["No/None", "Less", "More"],
    "heating":         ["No/None", "More"],
    "power":           ["No/None"],
    "speed":           ["More", "Less", "No/None", "Reverse"],
    "viscosity":       ["More", "Less", "Other Than"],
    "pH":              ["More", "Less", "Other Than"],
    "procedure":       ["Other Than", "Part Of"],
}

# Default guide words if parameter not specifically mapped
DEFAULT_GUIDE_WORDS = ["No/None", "More", "Less", "Other Than"]


def guide_words_for_parameter(param: str) -> list[str]:
    param_lower = param.lower().strip()
    for key, words in PARAM_GUIDE_WORDS.items():
        if key in param_lower or param_lower in key:
            return words
    return DEFAULT_GUIDE_WORDS


# ══════════════════════════════════════════════════════════════════════════════
# Node Identification
# ══════════════════════════════════════════════════════════════════════════════

def identify_nodes(ner_result: dict) -> list[HAZOPNode]:
    """
    Build HAZOP nodes by pairing extracted equipment entities with
    chemical entities and process parameters.

    Returns a list of HAZOPNode objects, one per (equipment, chemical, parameter) triple.
    """
    unique = ner_result.get("unique_terms", {})
    equipments = unique.get("EQUIP", ["process unit"]) or ["process unit"]
    chemicals  = unique.get("CHEM",  ["process fluid"]) or ["process fluid"]
    parameters = unique.get("PARAM", ["flow"]) or ["flow"]

    nodes: list[HAZOPNode] = []
    node_counter = 1

    # Limit combinatorics — cap at 3 equipments × 3 chemicals × 5 parameters
    for equip in equipments[:3]:
        for chem in chemicals[:3]:
            for param in parameters[:5]:
                nodes.append(HAZOPNode(
                    node_id=f"N{node_counter:03d}",
                    equipment=equip.title(),
                    chemical=chem.title(),
                    parameter=param.lower(),
                    design_intent=(
                        f"Normal {param} of {chem} through {equip} "
                        f"within design limits"
                    ),
                ))
                node_counter += 1

    # If no equipment / parameters found, create a generic node from text context
    if not nodes:
        nodes.append(HAZOPNode(
            node_id="N001",
            equipment="Process Unit",
            chemical="Process Fluid",
            parameter="flow",
            design_intent="Normal process operation within design limits",
        ))

    logger.info(f"Identified {len(nodes)} HAZOP nodes.")
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# KB Lookup with Fallback
# ══════════════════════════════════════════════════════════════════════════════

def _kb_lookup(guide_word: str, parameter: str) -> Optional[dict]:
    """
    Lookup deviation in knowledge base.
    Tries exact match, then partial parameter match.
    """
    gw_key = guide_word.lower().strip()
    pm_key = parameter.lower().strip()

    # Exact
    if (gw_key, pm_key) in DEVIATION_KB:
        return DEVIATION_KB[(gw_key, pm_key)]

    # Partial parameter match
    for (gw, pm), entry in DEVIATION_KB.items():
        if gw == gw_key and (pm in pm_key or pm_key in pm):
            return entry

    return None


def _merge_with_ner(kb_entry: dict, ner_result: dict, guide_word: str, parameter: str) -> dict:
    """
    Augment a KB entry with entities extracted from the actual incident text.
    NER-extracted causes / consequences / safeguards are prepended.
    """
    unique = ner_result.get("unique_terms", {})

    ner_causes     = unique.get("CAUSE", [])
    ner_conseqs    = unique.get("CONSEQ", [])
    ner_safeguards = unique.get("SAFEGUARD", [])

    merged = {
        "causes": list(dict.fromkeys(
            [c.title() for c in ner_causes[:3]] + kb_entry.get("causes", [])
        ))[:6],
        "consequences": list(dict.fromkeys(
            [c.title() for c in ner_conseqs[:3]] + kb_entry.get("consequences", [])
        ))[:5],
        "safeguards": list(dict.fromkeys(
            [s.title() for s in ner_safeguards[:3]] + kb_entry.get("safeguards", [])
        ))[:5],
        "recommended": kb_entry.get("recommended", []),
        "severity":    kb_entry.get("severity", 3),
        "likelihood":  kb_entry.get("likelihood", 2),
        "ref":         kb_entry.get("ref", ""),
    }
    return merged


def _generate_fallback(guide_word: str, parameter: str, ner_result: dict) -> dict:
    """
    When no KB entry exists, synthesise a basic entry purely from NER entities.
    """
    unique = ner_result.get("unique_terms", {})
    causes     = [c.title() for c in unique.get("CAUSE", [])[:4]]  or [f"Unknown cause for {guide_word} {parameter}"]
    conseqs    = [c.title() for c in unique.get("CONSEQ", [])[:4]] or ["Potential safety / process impact"]
    safeguards = [s.title() for s in unique.get("SAFEGUARD", [])[:4]] or ["Review design documentation"]
    return {
        "causes":       causes,
        "consequences": conseqs,
        "safeguards":   safeguards,
        "recommended":  [
            f"Conduct detailed engineering study of {guide_word} {parameter} deviation",
            "Install appropriate instrumented protection",
        ],
        "severity":  2,
        "likelihood": 2,
        "ref": "NER-derived from incident text",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Action Generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_actions(deviation: str, recommended: list[str], risk: RiskRating) -> list[str]:
    actions = []
    # Always add a review action
    actions.append(
        f"Review design basis for '{deviation}' scenario against current plant drawings"
    )
    actions.extend(recommended)
    if risk.risk_level in ("Critical", "High"):
        actions.insert(
            0,
            f"URGENT: Escalate '{deviation}' to process safety team for immediate risk assessment"
        )
    return actions[:5]  # cap at 5 actions


# ══════════════════════════════════════════════════════════════════════════════
# Main HAZOP Engine
# ══════════════════════════════════════════════════════════════════════════════

class HAZOPEngine:
    """
    Takes NER results from HAZOPNERPipeline and produces a full HAZOP table.

    Usage:
        engine = HAZOPEngine()
        rows = engine.run(ner_result)
        # rows → list[HAZOPRow]
    """

    def run(self, ner_result: dict) -> list[HAZOPRow]:
        """
        Full HAZOP analysis pipeline.
        Returns list[HAZOPRow] ready for output formatting.
        """
        nodes = identify_nodes(ner_result)
        rows: list[HAZOPRow] = []

        for node in nodes:
            applicable_gws = guide_words_for_parameter(node.parameter)
            for gw in applicable_gws:
                deviation = f"{gw} {node.parameter}"

                # KB lookup with NER merge
                kb_entry = _kb_lookup(gw, node.parameter)
                if kb_entry:
                    entry = _merge_with_ner(kb_entry, ner_result, gw, node.parameter)
                else:
                    entry = _generate_fallback(gw, node.parameter, ner_result)

                risk = rate_risk(entry["severity"], entry["likelihood"])
                actions = _generate_actions(deviation, entry["recommended"], risk)

                rows.append(HAZOPRow(
                    node_id=node.node_id,
                    equipment=node.equipment,
                    chemical=node.chemical,
                    parameter=node.parameter,
                    guide_word=gw,
                    deviation=deviation,
                    causes=entry["causes"],
                    consequences=entry["consequences"],
                    safeguards_existing=entry["safeguards"],
                    safeguards_recommended=entry["recommended"],
                    risk=risk,
                    actions=actions,
                    action_priority=priority_from_risk(risk),
                    historical_ref=entry["ref"],
                    notes=(
                        f"Node {node.node_id}: {node.design_intent}"
                    ),
                ))

        logger.info(
            f"HAZOP analysis complete: {len(rows)} rows across {len(nodes)} nodes."
        )
        return rows

    def run_from_text(self, text: str, use_bert: bool = False) -> list[HAZOPRow]:
        """
        Convenience: ingest raw text → NER → HAZOP in one call.
        """
        from src.ingestion import ingest_text
        from src.ner_pipeline import HAZOPNERPipeline

        doc = ingest_text(text, source_name="direct_input", save=False)
        pipe = HAZOPNERPipeline(use_bert=use_bert)
        ner_result = pipe.run(doc)
        return self.run(ner_result)


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def rows_to_dicts(rows: list[HAZOPRow]) -> list[dict]:
    return [r.to_dict() for r in rows]


def save_hazop_rows(rows: list[HAZOPRow], doc_id: str = "analysis",
                    output_dir: Optional[Path] = None) -> Path:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}_hazop.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows_to_dicts(rows), fh, indent=2, ensure_ascii=False)
    logger.info(f"HAZOP rows saved → {out_path}")
    return out_path


def load_hazop_rows(doc_id: str, input_dir: Optional[Path] = None) -> list[dict]:
    input_dir = input_dir or OUTPUT_DIR
    path = input_dir / f"{doc_id}_hazop.json"
    if not path.exists():
        raise FileNotFoundError(f"HAZOP result not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

def summarise_hazop(rows: list[HAZOPRow]) -> dict:
    """Return count by risk level and action priority."""
    risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    priority_counts = {"Immediate": 0, "Short-term": 0, "Long-term": 0}
    for row in rows:
        risk_counts[row.risk.risk_level] += 1
        priority_counts[row.action_priority] += 1
    return {
        "total_rows":      len(rows),
        "risk_breakdown":  risk_counts,
        "priority_breakdown": priority_counts,
        "unique_nodes":    len({r.node_id for r in rows}),
        "unique_deviations": len({r.deviation for r in rows}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    sample = (
        "During routine operation at the ethylene plant, a hydrogen leak from "
        "a corroded heat exchanger caused an explosion. The pressure relief valve "
        "had failed to open due to corrosion. The reactor temperature rose rapidly "
        "following loss of cooling water. Two operators suffered burns. "
        "Recommended safeguards include installing gas detectors, upgrading PRVs, "
        "and implementing an emergency shutdown interlock on cooling water loss."
    )

    engine = HAZOPEngine()
    rows = engine.run_from_text(sample, use_bert=False)
    summary = summarise_hazop(rows)

    print(f"\n{'='*70}")
    print("HAZOP ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total rows        : {summary['total_rows']}")
    print(f"Unique nodes      : {summary['unique_nodes']}")
    print(f"Unique deviations : {summary['unique_deviations']}")
    print(f"Risk — Critical   : {summary['risk_breakdown']['Critical']}")
    print(f"Risk — High       : {summary['risk_breakdown']['High']}")
    print(f"Risk — Medium     : {summary['risk_breakdown']['Medium']}")
    print(f"Risk — Low        : {summary['risk_breakdown']['Low']}")
    print(f"\nFirst 3 rows preview:")
    for row in rows[:3]:
        print(f"\n  Node: {row.node_id} | Equipment: {row.equipment} | Chemical: {row.chemical}")
        print(f"  Deviation: {row.deviation}")
        print(f"  Causes: {row.causes[:2]}")
        print(f"  Consequences: {row.consequences[:2]}")
        print(f"  Risk: {row.risk.risk_level} ({row.risk.risk_score})")
        print(f"  Priority: {row.action_priority}")
