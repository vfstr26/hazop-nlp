"""
lopa_engine.py — Layer of Protection Analysis (LOPA) probability engine.

Upgrades the heuristic 5×5 risk matrix to quantified risk using:
  - Initiating Event Frequencies (IEF) from CCPS LOPA data
  - Independent Protection Layer (IPL) Probability of Failure on Demand (PFD)
  - Consequence severity categories with tolerable risk targets
  - Mitigated Event Likelihood (MEL) calculation
  - Risk gap analysis: how many more IPL credits are needed

References:
  CCPS "Layer of Protection Analysis: Simplified Process Risk Assessment" (2001)
  IEC 61511 / ISA-84 Safety Instrumented Systems
  CCPS "Guidelines for Chemical Process Quantitative Risk Analysis" (2nd Ed.)

Output per scenario:
  - Unmitigated frequency (events/year)
  - Mitigated frequency after applying IPL credits
  - Tolerable frequency target
  - Risk gap (orders of magnitude short of target)
  - Recommended SIL level for any required SIS
  - LOPA worksheet row
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Optional
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Initiating Event Frequencies (events/year)
# Source: CCPS LOPA Table 3-1 and OREDA reliability data
# ══════════════════════════════════════════════════════════════════════════════

INITIATING_EVENT_FREQUENCIES: dict[str, float] = {
    # Control failures
    "control_valve_fails_open":         1e-1,
    "control_valve_fails_closed":       1e-1,
    "regulator_fails_open":             1e-1,
    "flow_controller_failure":          1e-1,

    # Instrument failures
    "instrument_failure_dangerous":     1e-1,
    "check_valve_failure":              1e-1,
    "operator_error_routine":           1e-1,
    "operator_error_non_routine":       1e-2,

    # Equipment failures
    "pump_seal_failure":                1e-1,
    "pump_loss_of_flow":                1e-1,
    "compressor_seal_failure":          1e-1,
    "heat_exchanger_tube_rupture":      1e-2,
    "vessel_overfill":                  1e-1,
    "pipe_leak_small":                  1e-2,
    "pipe_rupture_full_bore":           1e-4,
    "gasket_flange_failure":            1e-2,

    # Process deviations
    "cooling_water_loss":               1e-1,
    "steam_loss":                       1e-1,
    "power_failure":                    1e-1,
    "runaway_reaction_initiation":      1e-2,
    "contamination_wrong_chemical":     1e-2,
    "external_fire":                    1e-3,

    # Default
    "generic_process_upset":            1e-1,
    "generic_equipment_failure":        1e-2,
    "generic_human_error":              1e-1,
}

# Mapping from FailureMode strings to IEF keys
FAILURE_MODE_TO_IEF: dict[str, str] = {
    "Fails Closed":           "control_valve_fails_closed",
    "Fails Open":             "control_valve_fails_open",
    "Fails High":             "instrument_failure_dangerous",
    "Fails Low":              "instrument_failure_dangerous",
    "Spurious Trip / Activation": "instrument_failure_dangerous",
    "External Leak":          "pipe_leak_small",
    "Catastrophic Rupture":   "pipe_rupture_full_bore",
    "Plugs / Blocks":         "generic_process_upset",
    "Corrodes / Degrades":    "pipe_leak_small",
    "Overheats":              "cooling_water_loss",
    "Loses Power / Utility":  "power_failure",
    "Wrong / Erroneous Signal":"instrument_failure_dangerous",
    "Reverse Flow":           "check_valve_failure",
    "Excessive Vibration":    "generic_equipment_failure",
    "Human Error / Procedure Deviation": "operator_error_non_routine",
    "Catalyst Degradation":   "generic_process_upset",
    "Fouling / Scale Buildup":"generic_process_upset",
    "Cavitation":             "pump_loss_of_flow",
}


# ══════════════════════════════════════════════════════════════════════════════
# Independent Protection Layer PFDs
# Source: CCPS LOPA Table 4-1, IEC 61511
# ══════════════════════════════════════════════════════════════════════════════

IPL_PFD: dict[str, float] = {
    # Basic Process Control System (BPCS)
    "BPCS control loop":                    1e-1,
    "BPCS alarm with operator action":      1e-1,

    # Safety Instrumented System (SIS)
    "SIL 1 SIS":                            1e-1,
    "SIL 2 SIS":                            1e-2,
    "SIL 3 SIS":                            1e-3,

    # Passive safeguards
    "pressure relief valve (PSV/PRV)":      1e-2,
    "rupture disc":                         1e-2,
    "check valve":                          1e-1,
    "dike / bund (liquid release)":         1e-2,
    "flame arrester":                       1e-2,

    # Active safeguards
    "high pressure alarm + operator":       1e-1,
    "high temperature alarm + operator":    1e-1,
    "high level alarm + operator":          1e-1,
    "ESD system":                           1e-1,
    "emergency depressurisation":           1e-1,
    "automatic deluge / sprinkler":         1e-1,

    # Human intervention
    "operator response to DCS alarm":       1e-1,
    "trained operator emergency procedure": 1e-1,

    # Detection
    "gas detector + operator":              1e-1,
    "fire detector + emergency response":   1e-1,

    # Default
    "generic IPL":                          1e-1,
}

# Map safeguard keyword patterns to IPL keys
SAFEGUARD_TO_IPL: list[tuple[str, str]] = [
    ("PSV",                  "pressure relief valve (PSV/PRV)"),
    ("PRV",                  "pressure relief valve (PSV/PRV)"),
    ("relief valve",         "pressure relief valve (PSV/PRV)"),
    ("rupture disc",         "rupture disc"),
    ("SIS",                  "SIL 1 SIS"),
    ("ESD",                  "ESD system"),
    ("interlock",            "SIL 1 SIS"),
    ("PAHH",                 "SIL 1 SIS"),
    ("TAHH",                 "SIL 1 SIS"),
    ("LAHH",                 "SIL 1 SIS"),
    ("check valve",          "check valve"),
    ("NRV",                  "check valve"),
    ("alarm",                "BPCS alarm with operator action"),
    ("gas detector",         "gas detector + operator"),
    ("deluge",               "automatic deluge / sprinkler"),
    ("sprinkler",            "automatic deluge / sprinkler"),
    ("bund",                 "dike / bund (liquid release)"),
    ("dike",                 "dike / bund (liquid release)"),
    ("depressurisation",     "emergency depressurisation"),
    ("DCS",                  "BPCS control loop"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Consequence categories and tolerable frequencies
# Source: CCPS LOPA, typical process industry risk criteria
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsequenceCategory:
    name:                   str
    description:            str
    severity:               int      # 1–5
    tolerable_freq_per_year:float    # maximum acceptable mitigated event frequency

CONSEQUENCE_CATEGORIES = [
    ConsequenceCategory("Cat 1 — Minor",
        "No injury, minor damage, local release",
        1, 1e-1),
    ConsequenceCategory("Cat 2 — Serious",
        "Recordable injury, significant damage, restricted release",
        2, 1e-2),
    ConsequenceCategory("Cat 3 — Major",
        "Lost-time injury, major damage, community impact",
        3, 1e-3),
    ConsequenceCategory("Cat 4 — Catastrophic (onsite)",
        "Fatality, total loss of unit, large release",
        4, 1e-4),
    ConsequenceCategory("Cat 5 — Catastrophic (offsite)",
        "Multiple fatalities, major environmental damage",
        5, 1e-5),
]

def _get_consequence_category(severity: int) -> ConsequenceCategory:
    idx = max(0, min(4, severity - 1))
    return CONSEQUENCE_CATEGORIES[idx]


# ══════════════════════════════════════════════════════════════════════════════
# SIL determination
# ══════════════════════════════════════════════════════════════════════════════

def _required_sil(risk_reduction_needed: float) -> str:
    """
    Determine required SIL from the risk reduction factor needed.
    Risk reduction factor = unmitigated_freq / tolerable_freq (after other IPLs).
    """
    if risk_reduction_needed <= 1:
        return "No SIS required"
    elif risk_reduction_needed <= 10:
        return "SIL 1"
    elif risk_reduction_needed <= 100:
        return "SIL 2"
    elif risk_reduction_needed <= 1000:
        return "SIL 3"
    else:
        return "SIL 3 + redesign required"


# ══════════════════════════════════════════════════════════════════════════════
# LOPA result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LOPAResult:
    scenario_id:             str
    headline:                str

    # Frequencies
    initiating_event_freq:   float    # events/year
    initiating_event_desc:   str
    enabling_conditions_factor: float = 1.0

    # IPLs applied
    ipls_applied:            list[dict] = field(default_factory=list)
    total_pfd_credit:        float = 1.0   # product of all IPL PFDs

    # Outcomes
    unmitigated_freq:        float = 0.0   # before IPLs
    mitigated_freq:          float = 0.0   # after IPLs
    tolerable_freq:          float = 1e-4  # target
    risk_gap_orders:         float = 0.0   # log10(mitigated / tolerable) — positive = gap
    consequence_category:    str   = "Cat 4 — Catastrophic (onsite)"
    severity:                int   = 4

    # SIL recommendation
    additional_credit_needed: float = 0.0
    required_sil:            str   = "No SIS required"

    # Pass/fail
    tolerable:               bool  = False
    lopa_notes:              str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        status = "✅ TOLERABLE" if self.tolerable else "❌ NOT TOLERABLE"
        return (
            f"{status} | IE: {self.initiating_event_freq:.1e}/yr | "
            f"Mitigated: {self.mitigated_freq:.1e}/yr | "
            f"Target: {self.tolerable_freq:.1e}/yr | "
            f"Gap: {self.risk_gap_orders:+.1f} OOM | "
            f"Required SIL: {self.required_sil}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# IPL identifier
# ══════════════════════════════════════════════════════════════════════════════

def identify_ipls(safeguards: list[str]) -> list[dict]:
    """
    Match a list of safeguard strings to known IPLs with their PFDs.
    Returns list of {description, pfd, ipl_type} dicts.
    """
    ipls = []
    matched_keys = set()

    for sg in safeguards:
        sg_lower = sg.lower()
        for keyword, ipl_key in SAFEGUARD_TO_IPL:
            if keyword.lower() in sg_lower and ipl_key not in matched_keys:
                pfd = IPL_PFD.get(ipl_key, 1e-1)
                ipls.append({
                    "description":  sg,
                    "ipl_type":     ipl_key,
                    "pfd":          pfd,
                    "credit_oom":   round(-math.log10(pfd), 1),
                })
                matched_keys.add(ipl_key)
                break

    return ipls


def _get_ief(failure_mode_str: str) -> tuple[float, str]:
    """Get initiating event frequency for a failure mode."""
    key = FAILURE_MODE_TO_IEF.get(failure_mode_str, "generic_process_upset")
    return INITIATING_EVENT_FREQUENCIES.get(key, 1e-1), key


# ══════════════════════════════════════════════════════════════════════════════
# Main LOPA engine
# ══════════════════════════════════════════════════════════════════════════════

class LOPAEngine:
    """
    Performs LOPA on a CompoundScenario.

    For compound scenarios (depth > 1), the initiating event frequency is
    the product of individual event frequencies (simultaneous failures are
    inherently rarer).
    """

    def analyse(
        self,
        scenario,            # CompoundScenario
        safeguards: Optional[list[str]] = None,
        enabling_conditions_factor: float = 1.0,
    ) -> LOPAResult:
        """
        Run LOPA for one scenario.

        Parameters
        ----------
        scenario   : CompoundScenario
        safeguards : override safeguards list (uses scenario.existing_safeguards if None)
        enabling_conditions_factor : fraction of time the enabling condition exists (0–1)
        """
        # ── Initiating event frequency ────────────────────────────────────────
        if scenario.events:
            # For compound scenarios: product of individual IEFs
            # (conservative approximation of simultaneous failure probability)
            ief = 1.0
            ief_descs = []
            for ev in scenario.events:
                fm = ev.failure_mode.value if hasattr(ev.failure_mode, "value") else str(ev.failure_mode)
                freq, key = _get_ief(fm)
                ief *= freq
                ief_descs.append(f"{ev.node_tag}: {key} ({freq:.1e}/yr)")
            ief_desc = " × ".join(ief_descs[:3])
        else:
            ief      = 1e-1
            ief_desc = "generic_process_upset"

        # Apply enabling conditions factor
        unmitigated_freq = ief * enabling_conditions_factor

        # ── Consequence category ──────────────────────────────────────────────
        cat   = _get_consequence_category(scenario.severity)
        tol_f = cat.tolerable_freq_per_year

        # ── Identify and apply IPLs ───────────────────────────────────────────
        sg_list = safeguards or scenario.existing_safeguards or []
        ipls    = identify_ipls(sg_list)

        # Remove IPLs defeated by the scenario's safeguard gaps
        gap_keywords = [g.lower() for g in (scenario.safeguard_gaps or [])]
        active_ipls  = []
        for ipl in ipls:
            defeated = any(
                ipl["ipl_type"].lower() in gap or ipl["description"].lower() in gap
                for gap in gap_keywords
            )
            if not defeated:
                active_ipls.append(ipl)
            else:
                logger.debug(f"IPL defeated by scenario: {ipl['ipl_type']}")

        # Total PFD credit = product of individual IPL PFDs
        total_pfd = 1.0
        for ipl in active_ipls:
            total_pfd *= ipl["pfd"]

        mitigated_freq = unmitigated_freq * total_pfd

        # ── Risk gap ──────────────────────────────────────────────────────────
        if mitigated_freq > 0 and tol_f > 0:
            risk_gap = math.log10(mitigated_freq / tol_f)
        else:
            risk_gap = -99.0

        tolerable = risk_gap <= 0.0

        # ── SIL recommendation ────────────────────────────────────────────────
        if not tolerable:
            additional_credit = mitigated_freq / tol_f
            sil = _required_sil(additional_credit)
        else:
            additional_credit = 0.0
            sil = "No additional SIS required"

        result = LOPAResult(
            scenario_id=scenario.scenario_id,
            headline=scenario.headline or scenario.short_id(),
            initiating_event_freq=ief,
            initiating_event_desc=ief_desc,
            enabling_conditions_factor=enabling_conditions_factor,
            ipls_applied=active_ipls,
            total_pfd_credit=total_pfd,
            unmitigated_freq=unmitigated_freq,
            mitigated_freq=mitigated_freq,
            tolerable_freq=tol_f,
            risk_gap_orders=round(risk_gap, 2),
            consequence_category=cat.name,
            severity=scenario.severity,
            additional_credit_needed=round(additional_credit, 2),
            required_sil=sil,
            tolerable=tolerable,
            lopa_notes=(
                f"Compound depth {scenario.compound_depth}. "
                f"{len(active_ipls)} active IPLs of {len(ipls)} identified. "
                + (f"{len(ipls)-len(active_ipls)} IPL(s) defeated by scenario."
                   if len(ipls) > len(active_ipls) else "")
            ),
        )

        return result

    def analyse_set(
        self,
        scenario_set,
        top_n: int = 50,
    ) -> list[LOPAResult]:
        """Run LOPA on the top_n highest-risk scenarios."""
        results = []
        scenarios = sorted(
            scenario_set.scenarios,
            key=lambda s: -s.risk_score
        )[:top_n]

        for sc in scenarios:
            try:
                result = self.analyse(sc)
                results.append(result)
            except Exception as exc:
                logger.warning(f"LOPA failed for {sc.scenario_id}: {exc}")

        not_tolerable = sum(1 for r in results if not r.tolerable)
        logger.info(
            f"LOPA complete: {len(results)} scenarios analysed | "
            f"{not_tolerable} not tolerable"
        )
        return results


# ══════════════════════════════════════════════════════════════════════════════
# LOPA worksheet formatter
# ══════════════════════════════════════════════════════════════════════════════

def format_lopa_table(results: list[LOPAResult]) -> list[dict]:
    """Format LOPA results as flat dicts for DataFrame / table display."""
    rows = []
    for r in results:
        ipl_summary = " | ".join(
            f"{i['ipl_type']} (PFD={i['pfd']:.0e})"
            for i in r.ipls_applied[:3]
        ) or "None identified"

        rows.append({
            "Scenario":           r.headline[:70],
            "IE Freq (/yr)":      f"{r.initiating_event_freq:.2e}",
            "Unmitigated (/yr)":  f"{r.unmitigated_freq:.2e}",
            "IPLs":               ipl_summary,
            "Total PFD Credit":   f"{r.total_pfd_credit:.2e}",
            "Mitigated (/yr)":    f"{r.mitigated_freq:.2e}",
            "Target (/yr)":       f"{r.tolerable_freq:.2e}",
            "Risk Gap (OOM)":     f"{r.risk_gap_orders:+.1f}",
            "Consequence":        r.consequence_category,
            "Tolerable":          "✅ Yes" if r.tolerable else "❌ No",
            "Required SIL":       r.required_sil,
            "Notes":              r.lopa_notes[:80],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit widget
# ══════════════════════════════════════════════════════════════════════════════

def render_lopa_panel(scenario_set):
    """Render LOPA results in Streamlit."""
    import streamlit as st
    import pandas as pd

    st.subheader("📐 LOPA — Layer of Protection Analysis")
    st.markdown(
        "Quantified risk analysis using initiating event frequencies "
        "(CCPS data) and IPL PFD credits (IEC 61511)."
    )

    top_n = st.slider("Analyse top N scenarios by risk", 5, 100, 20)

    if st.button("▶ Run LOPA", type="primary"):
        with st.spinner("Running LOPA…"):
            engine  = LOPAEngine()
            results = engine.analyse_set(scenario_set, top_n=top_n)

        not_tol = sum(1 for r in results if not r.tolerable)
        sil3    = sum(1 for r in results if "SIL 3" in r.required_sil)
        sil2    = sum(1 for r in results if "SIL 2" in r.required_sil)
        sil1    = sum(1 for r in results if "SIL 1" in r.required_sil)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Analysed",       len(results))
        c2.metric("❌ Not Tolerable", not_tol)
        c3.metric("SIL 1 needed",   sil1)
        c4.metric("SIL 2 needed",   sil2)
        c5.metric("SIL 3 needed",   sil3)

        st.divider()

        rows = format_lopa_table(results)
        df   = pd.DataFrame(rows)

        # Colour-code the Tolerable column
        def colour_tolerable(val):
            return "color: green" if "Yes" in str(val) else "color: red; font-weight: bold"

        st.dataframe(
            df.style.applymap(colour_tolerable, subset=["Tolerable"]),
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        # Download
        csv_str = df.to_csv(index=False)
        st.download_button(
            "⬇ Download LOPA Table (CSV)",
            data=csv_str,
            file_name="lopa_results.csv",
            mime="text/csv",
        )
