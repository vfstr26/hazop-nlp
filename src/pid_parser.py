"""
pid_parser.py — P&ID Parser for HAZOP Scenario Generator.

Accepts three input forms:
  1. Free-text description  — engineer describes the plant in prose
  2. Structured JSON        — pre-defined node/stream schema
  3. CSV node table         — tag, name, type, chemicals, conditions

Outputs a PIDSystem object consumed by the scenario generator.

Two parsing strategies:
  A. Rule-based   — regex + gazetteer (always available, no API)
  B. LLM-assisted — sends text to LLM, gets back structured JSON
                    then validates and builds PIDSystem from it
"""

from __future__ import annotations

import re
import json
import csv
import io
from pathlib import Path
from typing import Optional
from loguru import logger

from src.scenario_models import (
    PIDSystem, PIDNode, PIDStream, Instrument,
    DesignCondition, NodeType
)


# ══════════════════════════════════════════════════════════════════════════════
# Node-type keyword mapping
# ══════════════════════════════════════════════════════════════════════════════

NODE_TYPE_KEYWORDS: dict[NodeType, list[str]] = {
    NodeType.REACTOR:        ["reactor", "rxr", "autoclave", "riser"],
    NodeType.VESSEL:         ["vessel", "drum", "tank", "surge drum", "knockout drum",
                              "accumulator", "receiver", "flash drum"],
    NodeType.COLUMN:         ["column", "tower", "distillation", "absorber column",
                              "stripper", "fractionator"],
    NodeType.HEAT_EXCHANGER: ["heat exchanger", "hx", "condenser", "reboiler",
                              "cooler", "heater", "trim cooler", "chiller"],
    NodeType.PUMP:           ["pump", "centrifugal pump", "positive displacement",
                              "metering pump"],
    NodeType.COMPRESSOR:     ["compressor", "blower", "fan", "ejector"],
    NodeType.VALVE:          ["valve", "control valve", "cv", "fcv", "prv", "psv",
                              "relief valve", "check valve", "isolation valve",
                              "rupture disc"],
    NodeType.SEPARATOR:      ["separator", "three-phase", "two-phase", "slug catcher",
                              "decanter", "cyclone"],
    NodeType.FURNACE:        ["furnace", "fired heater", "reformer", "boiler"],
    NodeType.STORAGE:        ["storage tank", "bullet", "sphere", "horton sphere",
                              "atmospheric tank", "floating roof"],
    NodeType.SCRUBBER:       ["scrubber", "absorber", "wash column", "quench"],
    NodeType.FILTER:         ["filter", "strainer", "screen", "coalescer"],
    NodeType.COMPRESSOR:     ["compressor", "blower", "fan"],
    NodeType.UTILITY:        ["cooling water", "steam", "nitrogen", "instrument air",
                              "plant air", "flare", "drain"],
}

# Instrument type keywords
INSTRUMENT_KEYWORDS = {
    "flow":        ["fi", "fic", "fit", "fah", "fal", "fall", "fsh", "fsl", "ft", "fc"],
    "pressure":    ["pi", "pic", "pit", "pah", "pal", "pahh", "pall", "psh", "psl",
                    "prv", "psv", "pt", "pc"],
    "temperature": ["ti", "tic", "tit", "tah", "tal", "tahh", "tall", "tsh", "tsl",
                    "tt", "tc"],
    "level":       ["li", "lic", "lit", "lah", "lal", "lahh", "lall", "lsh", "lsl",
                    "lt", "lc"],
    "analyser":    ["at", "ai", "aic", "qit", "qi"],
}

# Chemical name patterns
CHEMICAL_PATTERN = re.compile(
    r'\b(hydrogen[\s\-]?sulfide|hydrogen|ammonia|methane|propane|ethylene|'
    r'ethane|butane|naphtha|gasoline|crude\s+oil|natural\s+gas|chlorine|'
    r'benzene|toluene|xylene|steam|cooling\s+water|nitrogen|water|'
    r'hydrochloric\s+acid|sulfuric\s+acid|caustic|sodium\s+hydroxide|'
    r'carbon\s+dioxide|oxygen|acetylene|vinyl\s+chloride|'
    r'acrylonitrile|ethanol|methanol|LPG|LNG|condensate|raffinate|'
    r'catalyst|inhibitor|process\s+fluid|hydrocarbon)\b',
    re.IGNORECASE
)

# Numeric condition patterns
TEMP_PATTERN  = re.compile(r'(\d+[\.]?\d*)\s*[°]?[Cc]')
PRESS_PATTERN = re.compile(r'(\d+[\.]?\d*)\s*(?:barg|bara|psig|bar|kPa|MPa)')
FLOW_PATTERN  = re.compile(r'(\d+[\.]?\d*)\s*(?:kg/h|t/h|m3/h|l/min|gpm)')

# Tag patterns:  V-101, R-201, E-301, P-101A, FCV-101, PSV-202
TAG_PATTERN = re.compile(
    r'\b([A-Z]{1,4}-?\d{2,4}[A-Z]?)\b'
)

# Stream / line patterns
STREAM_PATTERN = re.compile(
    r'(?:from|outlet\s+of|discharge\s+of)\s+'
    r'([A-Z]{1,4}-?\d{2,4}[A-Z]?)'
    r'\s+(?:to|into|feeds?)\s+'
    r'([A-Z]{1,4}-?\d{2,4}[A-Z]?)',
    re.IGNORECASE
)

UTILITY_KEYWORDS = [
    "cooling water", "CW", "steam", "nitrogen", "N2",
    "instrument air", "IA", "plant air", "PA",
    "electrical", "power", "DCS", "PLC", "SIS",
    "fire water", "FW",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _detect_node_type(text: str) -> NodeType:
    text_lower = text.lower()
    for node_type, keywords in NODE_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return node_type
    return NodeType.OTHER


def _extract_chemicals(text: str) -> list[str]:
    found = CHEMICAL_PATTERN.findall(text)
    seen, result = set(), []
    for c in found:
        cl = c.lower().strip()
        if cl not in seen:
            seen.add(cl)
            result.append(c.strip().title())
    return result[:6]


def _extract_conditions(text: str) -> DesignCondition:
    cond = DesignCondition()
    temps = TEMP_PATTERN.findall(text)
    if temps:
        cond.temperature_c = float(temps[0])
        if len(temps) > 1:
            cond.design_temp_c = float(temps[-1])
    press = PRESS_PATTERN.findall(text)
    if press:
        cond.pressure_barg = float(press[0])
        if len(press) > 1:
            cond.design_press_barg = float(press[-1])
    flows = FLOW_PATTERN.findall(text)
    if flows:
        cond.flow_kgh = float(flows[0])
    if any(w in text.lower() for w in ["gas", "vapour", "vapor", "steam"]):
        cond.phase = "gas"
    elif any(w in text.lower() for w in ["liquid", "slurry"]):
        cond.phase = "liquid"
    elif any(w in text.lower() for w in ["two-phase", "mixed", "multiphase"]):
        cond.phase = "mixed"
    return cond


def _parse_instrument_tag(tag: str) -> Optional[Instrument]:
    tag_upper = tag.upper()
    for itype, prefixes in INSTRUMENT_KEYWORDS.items():
        for prefix in prefixes:
            if tag_upper.startswith(prefix.upper()):
                function = "indication"
                if any(x in tag_upper for x in ["AH", "AL", "SH", "SL"]):
                    function = "alarm/trip"
                elif "C" in tag_upper[len(prefix):len(prefix)+1]:
                    function = "control"
                return Instrument(
                    tag=tag,
                    type=itype,
                    function=function,
                    is_sis="SH" in tag_upper or "SL" in tag_upper,
                )
    return None


def _extract_utilities(text: str) -> list[str]:
    found = []
    for u in UTILITY_KEYWORDS:
        if u.lower() in text.lower():
            found.append(u)
    return list(dict.fromkeys(found))


# ══════════════════════════════════════════════════════════════════════════════
# Strategy A — Rule-based text parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_text(text: str, system_name: str = "Process System") -> PIDSystem:
    """
    Parse a free-text P&ID description into a PIDSystem.

    Handles two text formats:
      - Paragraph prose ("The feed drum V-101 receives propane at 5 barg...")
      - Bullet/line-per-node ("V-101: Feed surge drum, propane, 5 barg, 40°C")
    """
    system = PIDSystem(name=system_name, raw_text=text)
    system.utilities = _extract_utilities(text)

    # Split into sentences / lines for per-node parsing
    lines = re.split(r'[\n\r]+|(?<=[.!])\s+', text)
    lines = [l.strip() for l in lines if l.strip()]

    node_map: dict[str, PIDNode] = {}

    for line in lines:
        # Find all equipment tags in this line
        tags = TAG_PATTERN.findall(line)
        for tag in tags:
            if tag in node_map:
                continue
            # Derive name: text after the tag on same line
            name_match = re.search(
                re.escape(tag) + r'\s*[:\-–]?\s*([^,.\n]+)',
                line, re.IGNORECASE
            )
            raw_name = name_match.group(1).strip() if name_match else ""
            # Clean the name — strip trailing numbers / redundant info
            raw_name = re.sub(r'\s+at\s+.*$', '', raw_name, flags=re.IGNORECASE).strip()
            if not raw_name:
                raw_name = f"Equipment {tag}"

            node_type  = _detect_node_type(raw_name + " " + line)
            chemicals  = _extract_chemicals(line)
            conditions = _extract_conditions(line)

            # Extract instrument tags in same line
            inst_tags  = [t for t in TAG_PATTERN.findall(line)
                          if t != tag and any(
                              t.upper().startswith(p.upper())
                              for prefixes in INSTRUMENT_KEYWORDS.values()
                              for p in prefixes
                          )]
            instruments = [i for i in (_parse_instrument_tag(t) for t in inst_tags) if i]

            node = PIDNode(
                tag=tag,
                name=raw_name[:80],
                node_type=node_type,
                chemicals=chemicals,
                conditions=conditions,
                instruments=instruments,
                description=line[:200],
            )
            node_map[tag] = node

        # Extract streams mentioned in this line
        for m in STREAM_PATTERN.finditer(line):
            from_tag, to_tag = m.group(1), m.group(2)
            stream = PIDStream(
                stream_id=f"S-{from_tag}-{to_tag}",
                from_tag=from_tag,
                to_tag=to_tag,
                chemicals=_extract_chemicals(line),
                has_check_valve="check valve" in line.lower() or "nrv" in line.lower(),
                has_isolation="isolation" in line.lower() or "block valve" in line.lower(),
            )
            if not any(
                s.from_tag == from_tag and s.to_tag == to_tag
                for s in system.streams
            ):
                system.streams.append(stream)
                # Update node connectivity
                if from_tag in node_map:
                    if to_tag not in node_map[from_tag].downstream_tags:
                        node_map[from_tag].downstream_tags.append(to_tag)
                if to_tag in node_map:
                    if from_tag not in node_map[to_tag].upstream_tags:
                        node_map[to_tag].upstream_tags.append(from_tag)

    system.nodes = list(node_map.values())

    # If no streams were parsed from explicit "from/to" patterns,
    # infer a linear chain from node order
    if not system.streams and len(system.nodes) >= 2:
        for i in range(len(system.nodes) - 1):
            a = system.nodes[i]
            b = system.nodes[i + 1]
            stream = PIDStream(
                stream_id=f"S-{a.tag}-{b.tag}",
                from_tag=a.tag,
                to_tag=b.tag,
            )
            system.streams.append(stream)
            a.downstream_tags.append(b.tag)
            b.upstream_tags.append(a.tag)

    logger.info(
        f"Rule-based parse: {len(system.nodes)} nodes, "
        f"{len(system.streams)} streams, "
        f"{len(system.utilities)} utilities"
    )
    return system


# ══════════════════════════════════════════════════════════════════════════════
# Strategy B — LLM-assisted parser
# ══════════════════════════════════════════════════════════════════════════════

LLM_PARSE_PROMPT = """
You are a process engineering expert. Parse the following P&ID description into
structured JSON representing the process system.

**P&ID Description:**
{text}

Return ONLY valid JSON with this exact structure (no markdown fences):
{{
  "system_name": "...",
  "utilities": ["cooling water", "steam", ...],
  "nodes": [
    {{
      "tag": "V-101",
      "name": "Feed Surge Drum",
      "node_type": "Vessel / Tank",
      "chemicals": ["propane", "butane"],
      "temperature_c": 40,
      "pressure_barg": 5.0,
      "design_pressure_barg": 10.0,
      "phase": "liquid",
      "moc": "carbon steel",
      "safeguards": ["PSV-101", "LAH-101", "LIC-101"],
      "description": "..."
    }}
  ],
  "streams": [
    {{
      "stream_id": "S-01",
      "from_tag": "V-101",
      "to_tag": "P-101",
      "chemicals": ["propane"],
      "phase": "liquid",
      "has_check_valve": false
    }}
  ]
}}

Node type must be one of: {node_types}
Include every piece of equipment, instrument, and utility mentioned.
"""

NODE_TYPE_VALUES = [nt.value for nt in NodeType]


def parse_text_with_llm(text: str, llm_client=None,
                         system_name: str = "Process System") -> PIDSystem:
    """
    Use LLM to parse free-text P&ID description into structured PIDSystem.
    Falls back to rule-based parser if LLM fails.
    """
    if llm_client is None:
        from src.llm_inference import get_llm_client
        llm_client = get_llm_client()

    prompt = LLM_PARSE_PROMPT.format(
        text=text[:3000],
        node_types=", ".join(f'"{v}"' for v in NODE_TYPE_VALUES),
    )

    try:
        raw = llm_client.complete(prompt, max_tokens=2000)
        # Strip markdown fences
        raw = re.sub(r'^```[a-z]*\n?', '', raw.strip())
        raw = re.sub(r'\n?```$', '', raw)
        data = json.loads(raw)
    except Exception as exc:
        logger.warning(f"LLM parse failed ({exc}), falling back to rule-based.")
        return parse_text(text, system_name)

    system = PIDSystem(
        name=data.get("system_name", system_name),
        raw_text=text,
        utilities=data.get("utilities", []),
    )

    for nd in data.get("nodes", []):
        node_type_str = nd.get("node_type", "Other")
        node_type = next(
            (nt for nt in NodeType if nt.value == node_type_str),
            NodeType.OTHER
        )
        cond = DesignCondition(
            temperature_c=nd.get("temperature_c"),
            pressure_barg=nd.get("pressure_barg"),
            design_temp_c=nd.get("design_temperature_c"),
            design_press_barg=nd.get("design_pressure_barg"),
            mawp_barg=nd.get("design_pressure_barg"),
            phase=nd.get("phase", "liquid"),
            moc=nd.get("moc", ""),
        )
        node = PIDNode(
            tag=nd.get("tag", "?"),
            name=nd.get("name", ""),
            node_type=node_type,
            chemicals=nd.get("chemicals", []),
            conditions=cond,
            safeguards=nd.get("safeguards", []),
            description=nd.get("description", ""),
        )
        system.nodes.append(node)

    for sd in data.get("streams", []):
        stream = PIDStream(
            stream_id=sd.get("stream_id", ""),
            from_tag=sd.get("from_tag", ""),
            to_tag=sd.get("to_tag", ""),
            chemicals=sd.get("chemicals", []),
            phase=sd.get("phase", "liquid"),
            has_check_valve=sd.get("has_check_valve", False),
        )
        system.streams.append(stream)

    # Update connectivity
    node_map = {n.tag: n for n in system.nodes}
    for s in system.streams:
        if s.from_tag in node_map:
            node_map[s.from_tag].downstream_tags.append(s.to_tag)
        if s.to_tag in node_map:
            node_map[s.to_tag].upstream_tags.append(s.from_tag)

    logger.info(
        f"LLM parse: {len(system.nodes)} nodes, "
        f"{len(system.streams)} streams"
    )
    return system


# ══════════════════════════════════════════════════════════════════════════════
# Strategy C — Structured JSON input
# ══════════════════════════════════════════════════════════════════════════════

def parse_json(data: dict) -> PIDSystem:
    """Build PIDSystem directly from a pre-structured JSON dict."""
    system = PIDSystem(
        name=data.get("system_name", "Process System"),
        utilities=data.get("utilities", []),
        raw_text=data.get("description", ""),
    )
    for nd in data.get("nodes", []):
        node_type = next(
            (nt for nt in NodeType if nt.value == nd.get("node_type", "")),
            NodeType.OTHER
        )
        cond = DesignCondition(**{
            k: nd.get(k)
            for k in DesignCondition.__dataclass_fields__
            if k in nd
        })
        node = PIDNode(
            tag=nd["tag"],
            name=nd.get("name", ""),
            node_type=node_type,
            chemicals=nd.get("chemicals", []),
            conditions=cond,
            safeguards=nd.get("safeguards", []),
            description=nd.get("description", ""),
        )
        system.nodes.append(node)
    for sd in data.get("streams", []):
        system.streams.append(PIDStream(**{
            k: sd.get(k, v)
            for k, v in PIDStream.__dataclass_fields__.items()
        }))
    return system


def parse_json_file(path) -> PIDSystem:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_json(data)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy D — CSV node table
# ══════════════════════════════════════════════════════════════════════════════

def parse_csv(csv_text: str) -> PIDSystem:
    """
    Parse a CSV table with columns:
      tag, name, node_type, chemicals, temperature_c, pressure_barg,
      phase, moc, safeguards, upstream, downstream
    """
    system = PIDSystem()
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        node_type = next(
            (nt for nt in NodeType if nt.value.lower() == row.get("node_type", "").lower()),
            _detect_node_type(row.get("node_type", "") + " " + row.get("name", ""))
        )
        cond = DesignCondition(
            temperature_c=float(row["temperature_c"]) if row.get("temperature_c") else None,
            pressure_barg=float(row["pressure_barg"]) if row.get("pressure_barg") else None,
            phase=row.get("phase", "liquid"),
            moc=row.get("moc", ""),
        )
        chems = [c.strip() for c in row.get("chemicals", "").split(";") if c.strip()]
        safeguards = [s.strip() for s in row.get("safeguards", "").split(";") if s.strip()]
        upstream = [t.strip() for t in row.get("upstream", "").split(";") if t.strip()]
        downstream = [t.strip() for t in row.get("downstream", "").split(";") if t.strip()]

        node = PIDNode(
            tag=row.get("tag", ""),
            name=row.get("name", ""),
            node_type=node_type,
            chemicals=chems,
            conditions=cond,
            safeguards=safeguards,
            upstream_tags=upstream,
            downstream_tags=downstream,
        )
        system.nodes.append(node)

        # Build streams from connectivity columns
        for dn in downstream:
            stream = PIDStream(
                stream_id=f"S-{node.tag}-{dn}",
                from_tag=node.tag,
                to_tag=dn,
                chemicals=chems,
            )
            if not any(s.from_tag == node.tag and s.to_tag == dn for s in system.streams):
                system.streams.append(stream)

    logger.info(f"CSV parse: {len(system.nodes)} nodes, {len(system.streams)} streams")
    return system


# ══════════════════════════════════════════════════════════════════════════════
# Auto-detect and dispatch
# ══════════════════════════════════════════════════════════════════════════════

def parse(
    input_data,
    use_llm: bool = False,
    llm_client=None,
    system_name: str = "Process System",
) -> PIDSystem:
    """
    Universal entry point. Auto-detects input type and dispatches.
    input_data can be: str (text/CSV), dict (JSON), or Path (file).
    """
    if isinstance(input_data, Path) or (
        isinstance(input_data, str) and Path(input_data).exists()
    ):
        path = Path(input_data)
        if path.suffix == ".json":
            return parse_json_file(path)
        elif path.suffix == ".csv":
            return parse_csv(path.read_text(encoding="utf-8"))
        else:
            text = path.read_text(encoding="utf-8")
            return (parse_text_with_llm(text, llm_client, system_name)
                    if use_llm else parse_text(text, system_name))

    if isinstance(input_data, dict):
        return parse_json(input_data)

    if isinstance(input_data, str):
        # Check if it looks like CSV (has commas, starts with header-like row)
        first_line = input_data.strip().splitlines()[0] if input_data.strip() else ""
        if first_line.lower().startswith("tag,") or ",tag," in first_line.lower():
            return parse_csv(input_data)
        return (parse_text_with_llm(input_data, llm_client, system_name)
                if use_llm else parse_text(input_data, system_name))

    raise ValueError(f"Unsupported input type: {type(input_data)}")


# ══════════════════════════════════════════════════════════════════════════════
# Sample P&ID builder (for testing without real input)
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_pid() -> PIDSystem:
    """
    Builds a realistic propylene glycol reactor / distillation P&ID
    for use in demos and tests.
    """
    system = PIDSystem(
        name="Ethylene Oxide Hydration Unit",
        description="Production of ethylene glycol by hydration of ethylene oxide",
        utilities=["cooling water", "steam", "nitrogen", "instrument air", "DCS", "SIS"],
    )

    nodes = [
        PIDNode(
            tag="V-101", name="Ethylene Oxide Feed Drum",
            node_type=NodeType.VESSEL,
            chemicals=["Ethylene Oxide"],
            conditions=DesignCondition(
                temperature_c=20, pressure_barg=3.0,
                design_temp_c=60, design_press_barg=8.0,
                mawp_barg=7.0, phase="liquid", moc="stainless steel"
            ),
            safeguards=["PSV-101", "LAH-101", "LAL-101", "PAH-101", "gas detector"],
            description="Feed storage drum for liquid ethylene oxide",
            downstream_tags=["P-101"],
        ),
        PIDNode(
            tag="P-101", name="EO Feed Pump",
            node_type=NodeType.PUMP,
            chemicals=["Ethylene Oxide"],
            conditions=DesignCondition(
                temperature_c=20, pressure_barg=8.0,
                phase="liquid", moc="stainless steel"
            ),
            safeguards=["FAL-101", "spare pump P-101B", "check valve"],
            description="Centrifugal pump feeding EO to reactor",
            upstream_tags=["V-101"],
            downstream_tags=["R-201"],
        ),
        PIDNode(
            tag="R-201", name="Hydration Reactor",
            node_type=NodeType.REACTOR,
            chemicals=["Ethylene Oxide", "Water", "Ethylene Glycol"],
            conditions=DesignCondition(
                temperature_c=190, pressure_barg=20.0,
                design_temp_c=230, design_press_barg=30.0,
                mawp_barg=28.0, phase="liquid", moc="stainless steel"
            ),
            safeguards=[
                "PSV-201", "TAHH-201 (reactor ESD)", "FAL-201 (water flow)",
                "PAHH-201", "emergency depressurisation", "nitrogen purge",
            ],
            description="Tubular reactor — exothermic hydration of ethylene oxide with water",
            upstream_tags=["P-101", "P-102"],
            downstream_tags=["E-301"],
        ),
        PIDNode(
            tag="P-102", name="Process Water Pump",
            node_type=NodeType.PUMP,
            chemicals=["Water"],
            conditions=DesignCondition(
                temperature_c=25, pressure_barg=22.0,
                phase="liquid", moc="carbon steel"
            ),
            safeguards=["FAL-102", "spare pump P-102B"],
            description="High-pressure water feed pump to reactor",
            downstream_tags=["R-201"],
        ),
        PIDNode(
            tag="E-301", name="Reactor Effluent Cooler",
            node_type=NodeType.HEAT_EXCHANGER,
            chemicals=["Ethylene Glycol", "Water"],
            conditions=DesignCondition(
                temperature_c=90, pressure_barg=18.0,
                design_temp_c=230, design_press_barg=25.0,
                phase="liquid", moc="stainless steel"
            ),
            safeguards=["TAH-301", "CW flow alarm", "bypass valve"],
            description="Shell-and-tube cooler using cooling water on shell side",
            upstream_tags=["R-201"],
            downstream_tags=["V-301"],
        ),
        PIDNode(
            tag="V-301", name="Reactor Effluent Drum",
            node_type=NodeType.VESSEL,
            chemicals=["Ethylene Glycol", "Water"],
            conditions=DesignCondition(
                temperature_c=80, pressure_barg=2.0,
                design_temp_c=120, design_press_barg=5.0,
                mawp_barg=4.5, phase="liquid", moc="carbon steel"
            ),
            safeguards=["PSV-301", "LAH-301", "LAL-301", "LIC-301"],
            description="Intermediate drum — level control before distillation",
            upstream_tags=["E-301"],
            downstream_tags=["C-401"],
        ),
        PIDNode(
            tag="C-401", name="Glycol Distillation Column",
            node_type=NodeType.COLUMN,
            chemicals=["Ethylene Glycol", "Water"],
            conditions=DesignCondition(
                temperature_c=120, pressure_barg=0.3,
                design_temp_c=180, design_press_barg=2.0,
                mawp_barg=1.8, phase="mixed", moc="carbon steel"
            ),
            safeguards=[
                "PSV-401", "PAH-401", "TAH-401", "LAH-401 (sump)",
                "reflux flow alarm", "reboiler steam flow alarm",
            ],
            description="Distillation column separating water overhead from glycol bottoms",
            upstream_tags=["V-301"],
            downstream_tags=["E-402", "V-402"],
        ),
        PIDNode(
            tag="E-402", name="Overhead Condenser",
            node_type=NodeType.HEAT_EXCHANGER,
            chemicals=["Water"],
            conditions=DesignCondition(
                temperature_c=40, pressure_barg=0.3,
                phase="mixed", moc="carbon steel"
            ),
            safeguards=["CW flow alarm", "TAH-402"],
            description="Total condenser — cooling water condenses water vapour overhead",
            upstream_tags=["C-401"],
            downstream_tags=["V-402"],
        ),
        PIDNode(
            tag="V-402", name="Reflux Drum",
            node_type=NodeType.VESSEL,
            chemicals=["Water"],
            conditions=DesignCondition(
                temperature_c=40, pressure_barg=0.3,
                design_press_barg=2.0, mawp_barg=1.8,
                phase="liquid", moc="carbon steel"
            ),
            safeguards=["PSV-402", "LAH-402", "LAL-402"],
            description="Accumulator drum for column reflux and distillate",
            upstream_tags=["E-402"],
            downstream_tags=["P-401"],
        ),
        PIDNode(
            tag="P-401", name="Reflux Pump",
            node_type=NodeType.PUMP,
            chemicals=["Water"],
            conditions=DesignCondition(
                temperature_c=40, pressure_barg=1.5,
                phase="liquid", moc="carbon steel"
            ),
            safeguards=["FAL-401", "spare pump P-401B"],
            description="Returns reflux liquid to top of distillation column",
            upstream_tags=["V-402"],
            downstream_tags=["C-401"],
        ),
    ]

    streams = [
        PIDStream("S-01", "V-101", "P-101", ["Ethylene Oxide"], "liquid", has_check_valve=False),
        PIDStream("S-02", "P-101", "R-201", ["Ethylene Oxide"], "liquid", has_check_valve=True),
        PIDStream("S-03", "P-102", "R-201", ["Water"],          "liquid", has_check_valve=True),
        PIDStream("S-04", "R-201", "E-301", ["Ethylene Glycol", "Water"], "liquid"),
        PIDStream("S-05", "E-301", "V-301", ["Ethylene Glycol", "Water"], "liquid"),
        PIDStream("S-06", "V-301", "C-401", ["Ethylene Glycol", "Water"], "liquid"),
        PIDStream("S-07", "C-401", "E-402", ["Water"],          "gas"),
        PIDStream("S-08", "E-402", "V-402", ["Water"],          "mixed"),
        PIDStream("S-09", "V-402", "P-401", ["Water"],          "liquid"),
        PIDStream("S-10", "P-401", "C-401", ["Water"],          "liquid"),
    ]

    system.nodes   = nodes
    system.streams = streams
    return system


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pid = build_sample_pid()
    print(pid.summary())
    for node in pid.nodes:
        print(f"  {node.tag:8s} | {node.node_type.value:22s} | "
              f"{node.name:35s} | {', '.join(node.chemicals[:2])}")
