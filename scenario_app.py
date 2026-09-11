"""
scenario_app.py — HAZOP Scenario Generation Streamlit App.

Run:
    streamlit run scenario_app.py

Pages:
  1. 🏭 P&ID Input       — describe or upload your process design
  2. ⚙ Generate          — run the scenario engine with live progress
  3. 🔍 Scenario Browser  — filter, search, review every scenario
  4. 📊 Analytics         — risk heatmaps, node rankings, charts
  5. 📥 Export            — download all formats
  6. 🔗 Link to Incidents — cross-reference with hazop_nlp incident NER
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HAZOP Scenario Generator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.scenario_models    import ScenarioSet, CompoundScenario, RiskLevel, ScenarioStatus
from src.pid_parser         import build_sample_pid, parse, PIDSystem
from src.scenario_generator import ScenarioGenerator
from src.scenario_enricher  import ScenarioEnricher, mock_enrich_set
from src.scenario_ranker    import ScenarioRanker, get_statistics, cluster_by_consequence_type
from src.scenario_pipeline  import (
    run_pipeline, PipelineConfig, scenarios_to_hazop_rows,
    export_scenario_set, pid_from_ner,
)
from src.output_formatter   import to_html, to_json, to_csv, to_excel
from config                 import OUTPUT_DIR, SAMPLE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════════════════════

RISK_COLOURS = {
    "Critical": "#C0392B",
    "High":     "#E67E22",
    "Medium":   "#F1C40F",
    "Low":      "#27AE60",
}
RISK_TEXT = {
    "Critical": "white", "High": "white",
    "Medium": "#2C3E50",  "Low":  "white",
}

SAMPLE_PID_TEXT = """Ethylene oxide hydration unit producing ethylene glycol.

V-101 (EO Feed Drum): liquid ethylene oxide at 3 barg, 20°C. Safeguards: PSV-101, LAH-101, gas detector.
From V-101 to P-101 via centrifugal pump (EO Feed Pump). Check valve on discharge.
From P-101 to R-201 (Hydration Reactor): ethylene oxide + water at 190°C, 20 barg.
R-201 is an exothermic tubular reactor. Safeguards: PSV-201, TAHH-201 ESD interlock,
FAL-201 water flow alarm, emergency depressurisation, nitrogen purge.
P-102 (Process Water Pump) feeds water to R-201 at 22 barg.
R-201 outlet flows to E-301 (Reactor Effluent Cooler) using cooling water, then to
V-301 (Effluent Drum) at 2 barg. Safeguards: PSV-301, LAH-301, LIC-301.
V-301 feeds C-401 (Glycol Distillation Column) at 0.3 barg. Safeguards: PSV-401, TAH-401.
C-401 overhead goes to E-402 (Condenser) then V-402 (Reflux Drum).
V-402 feeds P-401 (Reflux Pump) back to C-401.
Utilities: cooling water, steam, nitrogen, instrument air, DCS, SIS.
"""

def _badge(text: str, bg: str, fg: str = "white", radius: str = "4px") -> str:
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:{radius};font-weight:700;font-size:0.8em">{text}</span>')

def _risk_badge(level: str) -> str:
    return _badge(level, RISK_COLOURS.get(level, "#95A5A6"),
                  RISK_TEXT.get(level, "white"))

def _depth_badge(depth: int) -> str:
    colours = {1: "#2980B9", 2: "#8E44AD", 3: "#C0392B", 4: "#C0392B"}
    return _badge(f"Depth-{depth}", colours.get(depth, "#7F8C8D"))

def _priority_badge(pri: str) -> str:
    c = {"Immediate": "#C0392B", "Short-term": "#E67E22", "Long-term": "#27AE60"}
    return _badge(pri, c.get(pri, "#7F8C8D"))

def _safe_list(v) -> list:
    return v if isinstance(v, list) else ([v] if v else [])


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "pid_text":       SAMPLE_PID_TEXT,
        "pid_system":     None,
        "scenario_set":   None,
        "generation_log": [],
        "stats":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏭 HAZOP Scenario Generator")
    st.caption("Generative AI for compounding failure analysis")
    st.divider()

    st.subheader("⚙ Generation Settings")

    max_depth = st.select_slider(
        "Max failure chain depth",
        options=[1, 2, 3],
        value=3,
        help="Depth-1 = single failures  |  Depth-2 = pairs  |  Depth-3 = triples",
    )
    max_scenarios = st.slider("Max raw scenarios", 100, 5000, 2000, step=100)
    top_n         = st.slider("Top N after ranking", 50, 500, 200, step=50)
    min_risk      = st.slider("Min risk score filter", 1, 10, 2)

    st.divider()
    st.subheader("🧠 LLM Enrichment")

    llm_provider = st.selectbox(
        "Provider",
        ["mock (offline)", "openai", "local (GGUF)"],
        help="'mock' requires no API key. 'openai' needs OPENAI_API_KEY in .env",
    )
    deep_n  = st.slider("Deep-enrich top N", 1, 30, 10)
    batch_n = st.slider("Batch-enrich next N", 0, 100, 40)

    use_llm_parser = st.toggle(
        "LLM P&ID parser",
        value=False,
        help="Use LLM to parse P&ID text (better extraction, uses API tokens)",
    )

    st.divider()
    st.caption("v1.0 · HAZOP NLP System")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_pid, tab_gen, tab_browse, tab_analytics, tab_export, tab_link = st.tabs([
    "🏭 P&ID Input",
    "⚙ Generate",
    "🔍 Scenario Browser",
    "📊 Analytics",
    "📥 Export",
    "🔗 Link to Incidents",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — P&ID INPUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_pid:
    st.header("🏭 P&ID Input")
    st.markdown(
        "Describe your process design in plain English — or paste a structured "
        "node/stream description. The parser extracts equipment, chemicals, "
        "instruments, and connectivity."
    )

    input_mode = st.radio(
        "Input method",
        ["Free-text description", "Upload JSON / CSV", "Use built-in sample"],
        horizontal=True,
    )

    if input_mode == "Use built-in sample":
        st.session_state["pid_text"] = SAMPLE_PID_TEXT
        st.info("Loaded: Ethylene Oxide Hydration Unit (10 nodes, 10 streams)")

    elif input_mode == "Upload JSON / CSV":
        uploaded = st.file_uploader("Upload P&ID file", type=["json", "csv", "txt"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            st.session_state["pid_text"] = content
            st.toast(f"Uploaded: {uploaded.name}", icon="📎")

    pid_text = st.text_area(
        "P&ID description",
        value=st.session_state.get("pid_text", ""),
        height=300,
        help="Include: equipment tags, names, chemicals, operating conditions, "
             "instruments, safeguards, and stream connections.",
    )
    st.session_state["pid_text"] = pid_text

    col_parse, col_sample = st.columns([2, 1])

    with col_parse:
        if st.button("🔍 Parse P&ID", type="primary", use_container_width=True):
            if not pid_text.strip():
                st.error("Enter a P&ID description first.")
            else:
                with st.spinner("Parsing P&ID…"):
                    provider_map = {
                        "mock (offline)": None,
                        "openai": "openai",
                        "local (GGUF)": "local",
                    }
                    llm = None
                    if use_llm_parser and llm_provider != "mock (offline)":
                        from src.llm_inference import get_llm_client
                        llm = get_llm_client(provider_map[llm_provider])

                    pid = parse(pid_text, use_llm=use_llm_parser and llm is not None,
                                llm_client=llm,
                                system_name="User Process System")
                    st.session_state["pid_system"] = pid

                st.success(
                    f"✅ Parsed: **{len(pid.nodes)} nodes** | "
                    f"**{len(pid.streams)} streams** | "
                    f"Utilities: {', '.join(pid.utilities[:4]) or 'none detected'}"
                )

    with col_sample:
        if st.button("Load Sample P&ID", use_container_width=True):
            pid = build_sample_pid()
            st.session_state["pid_system"] = pid
            st.session_state["pid_text"]   = SAMPLE_PID_TEXT
            st.toast("Sample P&ID loaded", icon="✅")

    # P&ID preview
    if st.session_state["pid_system"]:
        pid = st.session_state["pid_system"]
        st.divider()
        st.subheader("Parsed P&ID — Node Table")

        node_rows = []
        for n in pid.nodes:
            node_rows.append({
                "Tag":       n.tag,
                "Name":      n.name,
                "Type":      n.node_type.value,
                "Chemicals": ", ".join(n.chemicals[:3]),
                "Temp °C":   n.conditions.temperature_c or "—",
                "Press barg":n.conditions.pressure_barg or "—",
                "Safeguards":len(n.safeguards),
                "Downstream":len(n.downstream_tags),
            })

        import pandas as pd
        df = pd.DataFrame(node_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if pid.streams:
            with st.expander(f"Stream connections ({len(pid.streams)})"):
                stream_rows = [
                    {"Stream": s.stream_id, "From": s.from_tag, "To": s.to_tag,
                     "Phase": s.phase, "Check Valve": "✓" if s.has_check_valve else "—"}
                    for s in pid.streams
                ]
                st.dataframe(pd.DataFrame(stream_rows), use_container_width=True,
                             hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════

with tab_gen:
    st.header("⚙ Generate Scenarios")
    st.markdown(
        "The engine applies **all failure modes** to **every node**, then "
        "builds **compounding chains** up to depth-3 — the combinations a "
        "tired HAZOP team would never brainstorm. GenAI then enriches the "
        "highest-risk scenarios with detailed consequence analysis."
    )

    if not st.session_state["pid_system"]:
        st.info("👈 Parse a P&ID on the **P&ID Input** tab first.")
    else:
        pid = st.session_state["pid_system"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", len(pid.nodes))
        c2.metric("Streams", len(pid.streams))
        c3.metric("Est. raw scenarios",
                  f"~{len(pid.nodes) * 8 * (1 + len(pid.nodes) + len(pid.nodes)**2 // 6):,}")

        st.divider()

        if st.button("🚀 Generate Scenarios", type="primary", use_container_width=True):
            provider_map = {
                "mock (offline)": "mock",
                "openai": "openai",
                "local (GGUF)": "local",
            }
            provider = provider_map.get(llm_provider, "mock")

            cfg = PipelineConfig(
                max_depth=max_depth,
                max_scenarios=max_scenarios,
                min_risk_score=min_risk,
                llm_provider=provider,
                deep_enrich_n=deep_n,
                batch_enrich_n=batch_n,
                top_n=top_n,
                use_llm_parser=False,
            )

            log = []
            progress_bar = st.progress(0, text="Starting…")
            log_box = st.empty()

            def _progress(step: str, pct: int):
                progress_bar.progress(pct / 100, text=step)
                log.append(f"[{pct:3d}%] {step}")
                log_box.code("\n".join(log[-8:]))

            llm_client = None
            if provider != "mock":
                from src.llm_inference import get_llm_client
                llm_client = get_llm_client(provider, mock_fallback=True)

            ss = run_pipeline(
                pid,
                config=cfg,
                llm_client=llm_client,
                system_name=pid.name,
                progress_callback=_progress,
            )

            st.session_state["scenario_set"] = ss
            st.session_state["stats"]        = get_statistics(ss)
            progress_bar.progress(1.0, text="Complete ✓")

            st.success(
                f"✅ **{ss.total} scenarios** generated and ranked | "
                f"🔴 {ss.by_risk.get('Critical',0)} Critical  "
                f"🟠 {ss.by_risk.get('High',0)} High  "
                f"🟡 {ss.by_risk.get('Medium',0)} Medium  "
                f"🟢 {ss.by_risk.get('Low',0)} Low"
            )
            st.info("Explore results in the **Scenario Browser** and **Analytics** tabs →")

        # Show previous results if available
        if st.session_state["scenario_set"]:
            ss    = st.session_state["scenario_set"]
            stats = st.session_state["stats"] or {}
            st.divider()
            st.subheader("Generation Statistics")
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            r1.metric("Total",    ss.total)
            r2.metric("🔴 Critical", ss.by_risk.get("Critical", 0))
            r3.metric("🟠 High",     ss.by_risk.get("High", 0))
            r4.metric("🟡 Medium",   ss.by_risk.get("Medium", 0))
            r5.metric("🟢 Low",      ss.by_risk.get("Low", 0))
            r6.metric("🧠 Enriched", stats.get("llm_enriched_count", 0))

            with st.expander("Depth breakdown"):
                depth_data = ss.by_depth
                for d, cnt in sorted(depth_data.items()):
                    label = {
                        "1": "Depth-1 (single failures)",
                        "2": "Depth-2 (compounding pairs)",
                        "3": "Depth-3 (triple compounding)",
                    }.get(str(d), f"Depth-{d}")
                    st.markdown(f"**{label}:** {cnt:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIO BROWSER
# ══════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.header("🔍 Scenario Browser")

    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first on the **Generate** tab.")
    else:
        ss = st.session_state["scenario_set"]

        # ── Filter bar ────────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns(4)
        filter_risk = fc1.multiselect(
            "Risk Level",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
        filter_depth = fc2.multiselect(
            "Chain Depth",
            [1, 2, 3],
            default=[1, 2, 3],
            format_func=lambda d: f"Depth-{d}",
        )
        all_nodes = sorted({ev.node_tag for sc in ss.scenarios for ev in sc.events})
        filter_nodes = fc3.multiselect("Node Filter", all_nodes, default=[])
        search_text  = fc4.text_input("Search headline", placeholder="e.g. cooling, rupture…")

        filtered = ss.filter(
            risk_levels=filter_risk if filter_risk else None,
            min_depth=min(filter_depth) if filter_depth else 1,
        )
        if filter_depth:
            filtered = [s for s in filtered if s.compound_depth in filter_depth]
        if filter_nodes:
            filtered = [s for s in filtered
                        if any(ev.node_tag in filter_nodes for ev in s.events)]
        if search_text:
            filtered = [s for s in filtered
                        if search_text.lower() in s.headline.lower()
                        or any(search_text.lower() in ev.description.lower()
                               for ev in s.events)]

        st.caption(f"Showing **{len(filtered)}** of {ss.total} scenarios")
        st.divider()

        # ── Scenario cards ────────────────────────────────────────────────────
        if not filtered:
            st.warning("No scenarios match the current filters.")
        else:
            for sc in filtered[:100]:   # cap display at 100 for performance
                risk_level = sc.risk_level.value
                is_crit_high = risk_level in ("Critical", "High")

                with st.expander(
                    f"{_risk_badge(risk_level)} {_depth_badge(sc.compound_depth)}"
                    f"&nbsp; **{sc.short_id()}** — {sc.headline or sc.events[0].description if sc.events else ''}",
                    expanded=is_crit_high and len(filtered) <= 20,
                ):
                    col_chain, col_conseq, col_actions = st.columns([1.2, 1.2, 1])

                    with col_chain:
                        st.markdown("**🔗 Failure Chain**")
                        for i, ev in enumerate(sc.events, 1):
                            fm = ev.failure_mode.value if hasattr(ev.failure_mode, "value") else str(ev.failure_mode)
                            st.markdown(
                                f"`{i}.` **[{ev.node_tag}]** {fm}  \n"
                                f"&nbsp;&nbsp;&nbsp;&nbsp;→ *{ev.guide_word} {ev.parameter}*"
                            )
                        if sc.mechanism:
                            st.markdown("**🔬 Mechanism**")
                            st.caption(sc.mechanism)

                    with col_conseq:
                        st.markdown("**💥 Consequences**")
                        for c in _safe_list(sc.consequences)[:4]:
                            st.markdown(f"- {c}")
                        if sc.safeguard_gaps:
                            st.markdown("**⚠ Safeguard Gaps**")
                            for g in _safe_list(sc.safeguard_gaps)[:3]:
                                st.markdown(
                                    f'<span style="color:#C0392B">⚡ {g}</span>',
                                    unsafe_allow_html=True,
                                )

                    with col_actions:
                        st.markdown("**✅ Recommendations**")
                        for r in _safe_list(sc.recommendations)[:3]:
                            st.markdown(f"→ {r}")
                        if sc.historical_precedent:
                            st.markdown("**📖 Historical Precedent**")
                            st.caption(sc.historical_precedent)

                        # Risk chip
                        st.markdown(
                            f"**Risk:** {_risk_badge(risk_level)} "
                            f"**{sc.risk_score}/25** "
                            f"(S{sc.severity}×L{sc.likelihood})",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Priority:** {_priority_badge(sc.action_priority)}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Enriched:** {'✅ LLM' if sc.llm_enriched else '⚪ Rule-based'}"
                        )

                        # Status selector
                        new_status = st.selectbox(
                            "Status",
                            [s.value for s in ScenarioStatus],
                            index=0,
                            key=f"status_{sc.scenario_id}",
                            label_visibility="collapsed",
                        )
                        sc.status = ScenarioStatus(new_status)

            if len(filtered) > 100:
                st.info(f"Showing first 100 of {len(filtered)} — use filters to narrow down.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_analytics:
    st.header("📊 Analytics")

    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first.")
    else:
        ss    = st.session_state["scenario_set"]
        stats = get_statistics(ss)

        try:
            import pandas as pd

            # ── KPI row ───────────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Scenarios",  stats["total"])
            k2.metric("🔴 Critical",       stats["by_risk"].get("Critical", 0))
            k3.metric("⚠ Safeguard Defeats", stats["safeguard_defeats"])
            k4.metric("🧠 LLM Enriched",  stats["llm_enriched_count"])
            k5.metric("Depth-3 Compound", stats["by_depth"].get(3, 0))

            st.divider()
            row1, row2 = st.columns(2)

            # ── Risk by depth bar chart ───────────────────────────────────────
            with row1:
                st.subheader("Scenarios by Chain Depth")
                depth_df = pd.DataFrame([
                    {"Depth": f"Depth-{k}", "Count": v}
                    for k, v in sorted(stats["by_depth"].items())
                ])
                if not depth_df.empty:
                    st.bar_chart(depth_df.set_index("Depth"))

            # ── Consequence type pie ──────────────────────────────────────────
            with row2:
                st.subheader("Scenario Consequences")
                c_dist = stats.get("by_consequence_type", {})
                if c_dist:
                    c_df = pd.DataFrame(
                        [{"Type": k, "Count": v} for k, v in c_dist.items() if v > 0]
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(c_df.set_index("Type"))

            st.divider()
            row3, row4 = st.columns(2)

            # ── Top riskiest nodes ────────────────────────────────────────────
            with row3:
                st.subheader("Top Riskiest Nodes")
                node_risk = stats.get("top_riskiest_nodes", {})
                if node_risk:
                    nr_df = pd.DataFrame(
                        [{"Node": k, "Cumulative Risk Score": v}
                         for k, v in node_risk.items()]
                    )
                    st.bar_chart(nr_df.set_index("Node"))

            # ── Failure mode frequency ────────────────────────────────────────
            with row4:
                st.subheader("Most Common Failure Modes")
                fm_data = stats.get("failure_mode_freq", {})
                if fm_data:
                    fm_df = pd.DataFrame(
                        [{"Failure Mode": k, "Count": v} for k, v in fm_data.items()]
                    )
                    st.bar_chart(fm_df.set_index("Failure Mode"))

            st.divider()

            # ── Risk matrix heatmap ───────────────────────────────────────────
            st.subheader("5×5 Risk Matrix — Scenario Distribution")
            matrix_counts = [[0]*5 for _ in range(5)]
            for sc in ss.scenarios:
                s_idx = max(0, min(4, sc.severity   - 1))
                l_idx = max(0, min(4, sc.likelihood - 1))
                matrix_counts[4 - s_idx][l_idx] += 1

            matrix_df = pd.DataFrame(
                matrix_counts,
                index=[f"S{i} Severity" for i in range(5, 0, -1)],
                columns=[f"L{i} Likelihood" for i in range(1, 6)],
            )
            st.dataframe(
                matrix_df.style.background_gradient(cmap="RdYlGn_r"),
                use_container_width=True,
            )

            st.divider()

            # ── Full scenario table ───────────────────────────────────────────
            st.subheader("Scenario Summary Table")
            from src.scenario_ranker import ScenarioRanker
            ranker    = ScenarioRanker()
            tbl_rows  = ranker.get_summary_table(ss)
            tbl_df    = pd.DataFrame(tbl_rows)
            st.dataframe(tbl_df, use_container_width=True, hide_index=True,
                         height=400)

        except ImportError:
            st.warning("pandas not installed — charts unavailable. "
                       "Run: pip install pandas")
            st.json(stats)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("📥 Export Scenarios")

    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first.")
    else:
        ss    = st.session_state["scenario_set"]
        rows  = scenarios_to_hazop_rows(ss)
        stats = get_statistics(ss)

        summary = {
            "total_rows":       len(rows),
            "unique_nodes":     len(ss.by_node),
            "unique_deviations":len({r["deviation"] for r in rows}),
            "risk_breakdown":   ss.by_risk,
            "priority_breakdown": {
                "Immediate":  sum(1 for r in rows if r.get("action_priority") == "Immediate"),
                "Short-term": sum(1 for r in rows if r.get("action_priority") == "Short-term"),
                "Long-term":  sum(1 for r in rows if r.get("action_priority") == "Long-term"),
            },
        }
        meta = {
            "source": ss.system_name, "doc_id": ss.set_id,
            "date": ss.generated_at[:10], "chemicals": [],
        }

        st.markdown(
            f"**{ss.total} scenarios** ready to export for "
            f"**{ss.system_name}** (generated {ss.generated_at[:10]})"
        )

        ec1, ec2, ec3, ec4, ec5 = st.columns(5)

        # Raw ScenarioSet JSON
        with ec1:
            raw_json = json.dumps(ss.to_dict(), indent=2, default=str)
            st.download_button(
                "⬇ Scenarios JSON",
                data=raw_json,
                file_name=f"{ss.set_id}_scenarios.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Full scenario objects with all fields")

        # HAZOP HTML
        with ec2:
            html_str = to_html(rows, summary, meta)
            st.download_button(
                "⬇ HAZOP HTML",
                data=html_str,
                file_name=f"{ss.set_id}_hazop.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Print-ready coloured worksheet")

        # CSV
        with ec3:
            csv_str = to_csv(rows)
            st.download_button(
                "⬇ CSV",
                data=csv_str,
                file_name=f"{ss.set_id}_hazop.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Flat table for Excel / BI")

        # Excel
        with ec4:
            try:
                xl_bytes = to_excel(rows, summary, meta)
                st.download_button(
                    "⬇ Excel (.xlsx)",
                    data=xl_bytes,
                    file_name=f"{ss.set_id}_hazop.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption("Multi-sheet workbook")
            except Exception as e:
                st.warning(f"Excel: {e}")

        # Summary JSON
        with ec5:
            stats_json = json.dumps(stats, indent=2, default=str)
            st.download_button(
                "⬇ Statistics",
                data=stats_json,
                file_name=f"{ss.set_id}_stats.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Analytics summary")

        st.divider()
        st.subheader("Preview — HAZOP Worksheet")
        html_preview = to_html(rows[:50], summary, meta)
        st.components.v1.html(html_preview, height=600, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — LINK TO INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_link:
    st.header("🔗 Link to Historical Incidents")
    st.markdown(
        "Run the HAZOP NLP incident analysis on a real accident report "
        "and use the extracted entities to **seed the scenario generator** "
        "with real-world context — ensuring the generated scenarios match "
        "known failure patterns from history."
    )

    inc_text = st.text_area(
        "Paste incident report text",
        height=200,
        placeholder=(
            "Paste a CSB accident report, EPSC case study, or any incident text…\n\n"
            "The NER engine will extract chemicals, equipment, causes, and safeguards, "
            "then build a P&ID from the extracted entities for scenario generation."
        ),
    )

    c_load1, c_load2, c_load3 = st.columns(3)
    if c_load1.button("Load TX City 2005"):
        p = SAMPLE_DIR / "csb_texas_city_2005.txt"
        if p.exists():
            inc_text = p.read_text(encoding="utf-8")
            st.session_state["_inc_text"] = inc_text
    if c_load2.button("Load T2 Labs 2007"):
        p = SAMPLE_DIR / "csb_t2_laboratories_2007.txt"
        if p.exists():
            inc_text = p.read_text(encoding="utf-8")
            st.session_state["_inc_text"] = inc_text
    if c_load3.button("Load Ammonia Refrig."):
        p = SAMPLE_DIR / "ammonia_refrigeration_generic.txt"
        if p.exists():
            inc_text = p.read_text(encoding="utf-8")
            st.session_state["_inc_text"] = inc_text

    inc_text = inc_text or st.session_state.get("_inc_text", "")

    if st.button("🔬 Extract & Seed P&ID", type="primary") and inc_text.strip():
        with st.spinner("Running NER on incident text…"):
            from src.ingestion    import ingest_text
            from src.ner_pipeline import HAZOPNERPipeline

            doc        = ingest_text(inc_text, source_name="incident_link", save=False)
            pipe       = HAZOPNERPipeline(use_bert=False)
            ner_result = pipe.run(doc)

        unique = ner_result.get("unique_terms", {})
        st.success(
            f"✅ NER extracted: "
            f"{len(unique.get('CHEM',[]))} chemicals, "
            f"{len(unique.get('EQUIP',[]))} equipment, "
            f"{len(unique.get('CAUSE',[]))} causes, "
            f"{len(unique.get('SAFEGUARD',[]))} safeguards"
        )

        with st.spinner("Building P&ID from NER entities…"):
            pid = pid_from_ner(ner_result, system_name=doc["metadata"]["source"])
            st.session_state["pid_system"] = pid

        st.info(
            f"P&ID built: **{len(pid.nodes)} nodes**, **{len(pid.streams)} streams**. "
            "Go to the **Generate** tab to create scenarios →"
        )

        col_e, col_p = st.columns(2)
        with col_e:
            st.markdown("**Extracted entities preview:**")
            for label, terms in unique.items():
                if terms:
                    st.markdown(f"- **{label}:** {', '.join(t for t in terms[:5])}")
        with col_p:
            st.markdown("**Generated P&ID nodes:**")
            for n in pid.nodes[:8]:
                st.markdown(f"- `{n.tag}` {n.name} ({n.node_type.value})")
