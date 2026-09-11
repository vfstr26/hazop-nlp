"""
scenario_app.py — HAZOP Scenario Generation App (v2 — upgraded).

New in v2:
  - Authentication (login / roles: admin, engineer, viewer)
  - Persistent SQLite storage (sessions, scenarios, audit log)
  - P&ID image upload (vision LLM extracts nodes from scanned drawings)
  - Semantic KB search (embeddings over incident reports)
  - LOPA panel (quantified risk with IEF × IPL PFD)
  - Fine-tuned NER option (safety-domain BERT)

Run:
    streamlit run scenario_app.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="HAZOP Scenario Generator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate (renders login page if not authenticated) ───────────────────────
from src.auth import require_auth, render_user_menu, get_current_user, has_role, has_permission
require_auth("viewer")

from src.scenario_models    import ScenarioSet, ScenarioStatus, RiskLevel
from src.pid_parser         import build_sample_pid, parse, PIDSystem
from src.pid_image_parser   import PIDImageParser, render_pid_image_uploader
from src.scenario_generator import ScenarioGenerator
from src.scenario_enricher  import mock_enrich_set
from src.scenario_ranker    import ScenarioRanker, get_statistics
from src.scenario_pipeline  import (
    run_pipeline, PipelineConfig, scenarios_to_hazop_rows, export_scenario_set,
    pid_from_ner,
)
from src.semantic_kb        import get_semantic_kb
from src.lopa_engine        import LOPAEngine, render_lopa_panel, format_lopa_table
from src.database           import get_db
from src.output_formatter   import to_html, to_json, to_csv, to_excel
from config                 import OUTPUT_DIR, SAMPLE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

RISK_COLOURS = {
    "Critical": "#C0392B", "High": "#E67E22",
    "Medium": "#F1C40F",   "Low":  "#27AE60",
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

def _badge(text, bg, fg="white"):
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:4px;font-weight:700;font-size:0.8em">{text}</span>')

def _risk_badge(level):
    return _badge(level, RISK_COLOURS.get(level, "#95A5A6"), RISK_TEXT.get(level, "white"))

def _depth_badge(depth):
    c = {1: "#2980B9", 2: "#8E44AD", 3: "#C0392B"}
    return _badge(f"Depth-{depth}", c.get(depth, "#7F8C8D"))

def _priority_badge(pri):
    c = {"Immediate": "#C0392B", "Short-term": "#E67E22", "Long-term": "#27AE60"}
    return _badge(pri, c.get(pri, "#7F8C8D"))

def _safe_list(v):
    return v if isinstance(v, list) else ([v] if v else [])


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "pid_text":       SAMPLE_PID_TEXT,
        "pid_system":     None,
        "scenario_set":   None,
        "stats":          None,
        "db_session_id":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
db   = get_db()
user = get_current_user()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    render_user_menu()
    st.markdown("## 🏭 HAZOP Generator v2")
    st.divider()

    st.subheader("⚙ Generation Settings")
    max_depth     = st.select_slider("Max chain depth", [1, 2, 3], value=3)
    max_scenarios = st.slider("Max raw scenarios", 100, 5000, 2000, step=100)
    top_n         = st.slider("Top N after ranking", 50, 500, 200, step=50)
    min_risk      = st.slider("Min risk score", 1, 10, 2)

    st.divider()
    st.subheader("🧠 LLM Enrichment")
    llm_provider = st.selectbox("Provider",
        ["mock (offline)", "openai", "local (GGUF)"])
    deep_n  = st.slider("Deep-enrich top N", 1, 30, 10)
    batch_n = st.slider("Batch-enrich next N", 0, 100, 40)

    st.divider()
    st.subheader("🔬 NER Model")
    use_finetuned = st.toggle(
        "Fine-tuned safety NER",
        value=False,
        help="Uses safety-domain fine-tuned BERT. Run ner_trainer.py first to generate the model.",
    )

    st.divider()
    # Global stats
    g = db.global_stats()
    st.caption(
        f"📊 DB: {g['sessions']} sessions · {g['scenarios']:,} scenarios · "
        f"{g['review_pct']}% reviewed"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🏭 P&ID Input",
    "⚙ Generate",
    "🔍 Scenario Browser",
    "📐 LOPA",
    "🔎 Semantic Search",
    "📊 Analytics",
    "📥 Export",
    "💾 Sessions",
    "🔗 Link Incidents",
    "👥 Admin" if has_role("admin") else "ℹ About",
])

(tab_pid, tab_gen, tab_browse, tab_lopa, tab_search,
 tab_analytics, tab_export, tab_sessions, tab_link, tab_admin) = tabs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — P&ID INPUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_pid:
    st.header("🏭 P&ID Input")

    input_mode = st.radio(
        "Input method",
        ["Text description", "Upload image / PDF (Vision AI)", "Upload JSON / CSV", "Use sample"],
        horizontal=True,
    )

    if input_mode == "Use sample":
        st.session_state["pid_text"] = SAMPLE_PID_TEXT
        if st.button("Load Sample"):
            pid = build_sample_pid()
            st.session_state["pid_system"] = pid
            st.toast("Sample P&ID loaded ✅")

    elif input_mode == "Upload image / PDF (Vision AI)":
        st.info(
            "Upload a scanned P&ID drawing. GPT-4o vision will extract all nodes, "
            "streams, and instruments automatically. Requires OPENAI_API_KEY."
        )
        from src.llm_inference import get_llm_client
        api_key = ""
        try:
            import os; from dotenv import load_dotenv; load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY", "")
        except Exception:
            pass

        image_parser = PIDImageParser(api_key=api_key, model="gpt-4o")
        pid_from_image = render_pid_image_uploader(image_parser)
        if pid_from_image:
            st.session_state["pid_system"] = pid_from_image
            st.session_state["pid_text"]   = pid_from_image.description

    elif input_mode == "Upload JSON / CSV":
        uploaded = st.file_uploader("Upload P&ID file", type=["json", "csv", "txt"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            st.session_state["pid_text"] = content

    # Text area (always shown for text mode, shown as fallback otherwise)
    if input_mode in ("Text description", "Use sample"):
        pid_text = st.text_area(
            "P&ID description",
            value=st.session_state.get("pid_text", ""),
            height=280,
        )
        st.session_state["pid_text"] = pid_text

        col_parse, col_llm = st.columns([2, 1])
        use_llm_parser = col_llm.toggle("Use LLM parser", value=False)

        if col_parse.button("🔍 Parse P&ID", type="primary", use_container_width=True):
            with st.spinner("Parsing…"):
                pid = parse(
                    st.session_state["pid_text"],
                    use_llm=use_llm_parser,
                    system_name="User Process System",
                )
                st.session_state["pid_system"] = pid
            st.success(
                f"✅ {len(pid.nodes)} nodes · {len(pid.streams)} streams · "
                f"Utilities: {', '.join(pid.utilities[:4]) or 'none'}"
            )

    # Node table preview
    if st.session_state["pid_system"]:
        pid = st.session_state["pid_system"]
        st.divider()
        import pandas as pd
        node_rows = [{
            "Tag": n.tag, "Name": n.name, "Type": n.node_type.value,
            "Chemicals": ", ".join(n.chemicals[:2]),
            "T °C": n.conditions.temperature_c or "—",
            "P barg": n.conditions.pressure_barg or "—",
            "Safeguards": len(n.safeguards),
        } for n in pid.nodes]
        st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════

with tab_gen:
    st.header("⚙ Generate Scenarios")

    if not st.session_state["pid_system"]:
        st.info("Parse a P&ID on the **P&ID Input** tab first.")
    else:
        if not has_permission("run_analysis"):
            st.warning("Your role (viewer) cannot run analysis. Contact an engineer.")
        else:
            pid = st.session_state["pid_system"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Nodes",   len(pid.nodes))
            c2.metric("Streams", len(pid.streams))
            c3.metric("Est. scenarios",
                      f"~{len(pid.nodes) * 8 * (1 + len(pid.nodes)):,}")

            session_name = st.text_input(
                "Session name (for saving to DB)",
                value=f"{pid.name} — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            )

            if st.button("🚀 Generate Scenarios", type="primary", use_container_width=True):
                provider_map = {
                    "mock (offline)": "mock",
                    "openai": "openai",
                    "local (GGUF)": "local",
                }
                provider = provider_map.get(llm_provider, "mock")
                cfg = PipelineConfig(
                    max_depth=max_depth, max_scenarios=max_scenarios,
                    min_risk_score=min_risk, llm_provider=provider,
                    deep_enrich_n=deep_n, batch_enrich_n=batch_n, top_n=top_n,
                )

                log = []
                pbar    = st.progress(0, "Starting…")
                log_box = st.empty()

                def _progress(step, pct):
                    pbar.progress(pct / 100, step)
                    log.append(f"[{pct:3d}%] {step}")
                    log_box.code("\n".join(log[-6:]))

                llm_client = None
                if provider != "mock":
                    from src.llm_inference import get_llm_client
                    llm_client = get_llm_client(provider, mock_fallback=True)

                ss = run_pipeline(
                    pid, config=cfg, llm_client=llm_client,
                    system_name=pid.name, progress_callback=_progress,
                )
                st.session_state["scenario_set"] = ss
                st.session_state["stats"]        = get_statistics(ss)

                # Persist to DB
                sess_id = ss.set_id
                db.save_session(
                    sess_id, session_name, pid.name,
                    st.session_state.get("pid_text", ""),
                    created_by=user.get("username", "unknown"),
                )
                db.save_scenarios(sess_id, ss)
                db.log_action(user.get("username", ""), "generate", "session", sess_id,
                              f"{ss.total} scenarios")
                st.session_state["db_session_id"] = sess_id
                pbar.progress(1.0, "Complete ✓")

                st.success(
                    f"✅ **{ss.total} scenarios** | "
                    f"🔴 {ss.by_risk.get('Critical',0)} Critical · "
                    f"🟠 {ss.by_risk.get('High',0)} High · "
                    f"🟡 {ss.by_risk.get('Medium',0)} Medium · "
                    f"🟢 {ss.by_risk.get('Low',0)} Low"
                )

        if st.session_state["scenario_set"]:
            ss = st.session_state["scenario_set"]
            st.divider()
            r1,r2,r3,r4,r5,r6 = st.columns(6)
            r1.metric("Total",     ss.total)
            r2.metric("🔴 Critical", ss.by_risk.get("Critical",0))
            r3.metric("🟠 High",     ss.by_risk.get("High",0))
            r4.metric("🟡 Medium",   ss.by_risk.get("Medium",0))
            r5.metric("🟢 Low",      ss.by_risk.get("Low",0))
            r6.metric("🧠 Enriched", (st.session_state.get("stats") or {}).get("llm_enriched_count",0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIO BROWSER
# ══════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.header("🔍 Scenario Browser")

    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first.")
    else:
        ss = st.session_state["scenario_set"]

        fc1,fc2,fc3,fc4 = st.columns(4)
        filter_risk  = fc1.multiselect("Risk", ["Critical","High","Medium","Low"],
                                       default=["Critical","High"])
        filter_depth = fc2.multiselect("Depth", [1,2,3], default=[1,2,3],
                                       format_func=lambda d: f"Depth-{d}")
        all_nodes    = sorted({ev.node_tag for sc in ss.scenarios for ev in sc.events})
        filter_nodes = fc3.multiselect("Nodes", all_nodes)
        search_text  = fc4.text_input("Search", placeholder="cooling, rupture…")

        filtered = ss.filter(
            risk_levels=filter_risk or None,
            min_depth=min(filter_depth) if filter_depth else 1,
        )
        if filter_depth:
            filtered = [s for s in filtered if s.compound_depth in filter_depth]
        if filter_nodes:
            filtered = [s for s in filtered
                        if any(ev.node_tag in filter_nodes for ev in s.events)]
        if search_text:
            q = search_text.lower()
            filtered = [s for s in filtered
                        if q in s.headline.lower()
                        or any(q in ev.description.lower() for ev in s.events)]

        st.caption(f"Showing **{len(filtered)}** of {ss.total} scenarios")
        st.divider()

        for sc in filtered[:100]:
            level = sc.risk_level.value
            with st.expander(
                f"{_risk_badge(level)} {_depth_badge(sc.compound_depth)}"
                f"&nbsp; **{sc.short_id()}** — {sc.headline or (sc.events[0].description if sc.events else '')}",
                expanded=level == "Critical" and len(filtered) <= 10,
            ):
                c1,c2,c3 = st.columns([1.2,1.2,1])
                with c1:
                    st.markdown("**🔗 Failure Chain**")
                    for i, ev in enumerate(sc.events, 1):
                        fm = ev.failure_mode.value if hasattr(ev.failure_mode,"value") else str(ev.failure_mode)
                        st.markdown(f"`{i}.` **[{ev.node_tag}]** {fm}  \n&nbsp;&nbsp;&nbsp;→ *{ev.guide_word} {ev.parameter}*")
                    if sc.mechanism:
                        st.caption(sc.mechanism)

                with c2:
                    st.markdown("**💥 Consequences**")
                    for c in _safe_list(sc.consequences)[:3]:
                        st.markdown(f"- {c}")
                    if sc.safeguard_gaps:
                        st.markdown("**⚠ Safeguard Gaps**")
                        for g in _safe_list(sc.safeguard_gaps)[:2]:
                            st.markdown(f'<span style="color:#C0392B">⚡ {g}</span>',
                                        unsafe_allow_html=True)

                with c3:
                    st.markdown("**✅ Recommendations**")
                    for r in _safe_list(sc.recommendations)[:3]:
                        st.markdown(f"→ {r}")
                    if sc.historical_precedent:
                        st.caption(f"📖 {sc.historical_precedent}")
                    st.markdown(
                        f"**Risk:** {_risk_badge(level)} **{sc.risk_score}/25**  \n"
                        f"**Priority:** {_priority_badge(sc.action_priority)}",
                        unsafe_allow_html=True,
                    )

                    # Status update — engineers and admins only
                    if has_permission("review_scenarios"):
                        new_status = st.selectbox(
                            "Status",
                            [s.value for s in ScenarioStatus],
                            key=f"st_{sc.scenario_id}",
                            label_visibility="collapsed",
                        )
                        if new_status != sc.status.value:
                            sc.status = ScenarioStatus(new_status)
                            if st.session_state.get("db_session_id"):
                                db.update_scenario_status(
                                    sc.scenario_id, new_status,
                                    reviewer=user.get("username",""),
                                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOPA
# ══════════════════════════════════════════════════════════════════════════════

with tab_lopa:
    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first.")
    else:
        render_lopa_panel(st.session_state["scenario_set"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔎 Semantic Incident Search")
    st.markdown(
        "Search the indexed incident report library using natural language. "
        "The semantic KB uses sentence embeddings — no exact keyword match needed."
    )

    query = st.text_input(
        "Search query",
        placeholder="e.g. cooling water failure reactor temperature runaway",
    )

    col_build, col_search = st.columns([1, 3])
    filter_type = col_build.selectbox("Filter type",
        ["all", "cause", "consequence", "safeguard"])

    if col_build.button("🔨 Build / Refresh Index"):
        with st.spinner("Building semantic KB index…"):
            try:
                kb = get_semantic_kb(auto_build=False)
                n  = kb.build(force_rebuild=True)
                st.success(f"Index built: {n:,} chunks indexed.")
            except Exception as e:
                st.error(f"Index build failed: {e}  \nRun: `pip install sentence-transformers faiss-cpu`")

    if st.button("🔍 Search", type="primary") and query.strip():
        with st.spinner("Searching…"):
            try:
                kb   = get_semantic_kb()
                hits = kb.search(
                    query,
                    top_k=10,
                    filter_type=None if filter_type == "all" else filter_type,
                )
                if not hits:
                    st.warning("No results. Build the index first using the button above.")
                else:
                    for i, h in enumerate(hits, 1):
                        score_colour = "#27AE60" if h["score"] > 0.7 else (
                            "#E67E22" if h["score"] > 0.5 else "#7F8C8D")
                        st.markdown(
                            f'**{i}.** [{h["source"]}] '
                            f'<span style="background:{score_colour};color:white;'
                            f'padding:1px 8px;border-radius:4px;font-size:0.8em">'
                            f'{h["score"]:.0%}</span>  \n{h["text"]}',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Search failed: {e}  \nRun: `pip install sentence-transformers faiss-cpu`")

    # Similar incidents for current scenario set
    if st.session_state["scenario_set"]:
        st.divider()
        st.subheader("Historical Precedents for Top Scenarios")
        ss = st.session_state["scenario_set"]
        top3 = ss.top_scenarios(3)
        for sc in top3:
            with st.expander(f"{sc.short_id()} — {sc.headline[:80]}"):
                try:
                    kb   = get_semantic_kb()
                    hits = kb.find_similar_incidents(sc.headline, top_k=3)
                    if hits:
                        for h in hits:
                            st.markdown(f"- **[{h['score']:.0%}]** [{h['source']}] {h['text'][:150]}")
                    else:
                        st.caption("No indexed incidents found. Build the index first.")
                except Exception as e:
                    st.caption(f"Semantic KB unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ANALYTICS
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
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Total",              stats["total"])
            k2.metric("🔴 Critical",         stats["by_risk"].get("Critical",0))
            k3.metric("⚠ Safeguard Defeats", stats["safeguard_defeats"])
            k4.metric("🧠 LLM Enriched",    stats["llm_enriched_count"])
            k5.metric("Depth-3",            stats["by_depth"].get(3,0))
            st.divider()

            r1,r2 = st.columns(2)
            with r1:
                st.subheader("By Chain Depth")
                d_df = pd.DataFrame([
                    {"Depth": f"Depth-{k}", "Count": v}
                    for k,v in sorted(stats["by_depth"].items())
                ])
                if not d_df.empty:
                    st.bar_chart(d_df.set_index("Depth"))

            with r2:
                st.subheader("By Consequence Type")
                c_df = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k,v in stats.get("by_consequence_type",{}).items() if v > 0
                ]).sort_values("Count", ascending=False)
                if not c_df.empty:
                    st.bar_chart(c_df.set_index("Type"))

            st.divider()
            r3,r4 = st.columns(2)
            with r3:
                st.subheader("Top Riskiest Nodes")
                nr_df = pd.DataFrame([
                    {"Node": k, "Risk Score": v}
                    for k,v in stats.get("top_riskiest_nodes",{}).items()
                ])
                if not nr_df.empty:
                    st.bar_chart(nr_df.set_index("Node"))

            with r4:
                st.subheader("Failure Mode Frequency")
                fm_df = pd.DataFrame([
                    {"Failure Mode": k, "Count": v}
                    for k,v in stats.get("failure_mode_freq",{}).items()
                ])
                if not fm_df.empty:
                    st.bar_chart(fm_df.set_index("Failure Mode"))

            st.divider()
            st.subheader("5×5 Risk Matrix — Scenario Distribution")
            mat = [[0]*5 for _ in range(5)]
            for sc in ss.scenarios:
                s = max(0, min(4, sc.severity   - 1))
                l = max(0, min(4, sc.likelihood - 1))
                mat[4-s][l] += 1
            mat_df = pd.DataFrame(
                mat,
                index  =[f"S{i}" for i in range(5,0,-1)],
                columns=[f"L{i}" for i in range(1,6)],
            )
            st.dataframe(
                mat_df.style.background_gradient(cmap="RdYlGn_r"),
                use_container_width=True,
            )

        except ImportError:
            st.warning("pip install pandas")
            st.json(stats)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("📥 Export")

    if not st.session_state["scenario_set"]:
        st.info("Generate scenarios first.")
    else:
        ss    = st.session_state["scenario_set"]
        rows  = scenarios_to_hazop_rows(ss)
        stats = get_statistics(ss)
        summary = {
            "total_rows": len(rows),
            "unique_nodes": len(ss.by_node),
            "unique_deviations": len({r["deviation"] for r in rows}),
            "risk_breakdown": ss.by_risk,
            "priority_breakdown": {
                "Immediate":  sum(1 for r in rows if r.get("action_priority")=="Immediate"),
                "Short-term": sum(1 for r in rows if r.get("action_priority")=="Short-term"),
                "Long-term":  sum(1 for r in rows if r.get("action_priority")=="Long-term"),
            },
        }
        meta = {"source": ss.system_name, "doc_id": ss.set_id,
                "date": ss.generated_at[:10], "chemicals": []}

        e1,e2,e3,e4,e5 = st.columns(5)
        with e1:
            st.download_button("⬇ Scenarios JSON",
                data=json.dumps(ss.to_dict(), indent=2, default=str),
                file_name=f"{ss.set_id}_scenarios.json", mime="application/json",
                use_container_width=True)
            st.caption("Full scenario objects")
        with e2:
            st.download_button("⬇ HAZOP HTML",
                data=to_html(rows, summary, meta),
                file_name=f"{ss.set_id}_hazop.html", mime="text/html",
                use_container_width=True)
            st.caption("Print-ready worksheet")
        with e3:
            st.download_button("⬇ CSV",
                data=to_csv(rows),
                file_name=f"{ss.set_id}_hazop.csv", mime="text/csv",
                use_container_width=True)
            st.caption("Flat table")
        with e4:
            try:
                xl = to_excel(rows, summary, meta)
                st.download_button("⬇ Excel",
                    data=xl,
                    file_name=f"{ss.set_id}_hazop.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
                st.caption("Multi-sheet workbook")
            except Exception as e:
                st.warning(f"Excel: {e}")
        with e5:
            # LOPA CSV
            try:
                engine = LOPAEngine()
                lopa_results = engine.analyse_set(ss, top_n=min(50, ss.total))
                lopa_rows = format_lopa_table(lopa_results)
                import pandas as pd
                lopa_csv = pd.DataFrame(lopa_rows).to_csv(index=False)
                st.download_button("⬇ LOPA CSV",
                    data=lopa_csv,
                    file_name=f"{ss.set_id}_lopa.csv", mime="text/csv",
                    use_container_width=True)
                st.caption("LOPA worksheet")
            except Exception as e:
                st.warning(f"LOPA: {e}")

        st.divider()
        st.subheader("HTML Preview")
        st.components.v1.html(to_html(rows[:30], summary, meta), height=500, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_sessions:
    st.header("💾 Saved Sessions")

    sessions = db.list_sessions(
        created_by=None if has_role("admin") else user.get("username")
    )

    if not sessions:
        st.info("No saved sessions yet. Run a generation to create one.")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame([{
                "Session ID": s["session_id"][:8],
                "Name":       s["name"],
                "System":     s["system_name"],
                "Scenarios":  s["scenario_count"],
                "Created by": s["created_by"],
                "Date":       datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d %H:%M")
                               if s.get("created_at") else "—",
            } for s in sessions])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except ImportError:
            for s in sessions:
                st.write(f"- {s['session_id'][:8]} | {s['name']} | {s['scenario_count']} scenarios")

    st.divider()
    st.subheader("Audit Log")
    log = db.get_audit_log(limit=30)
    if log:
        try:
            import pandas as pd
            log_df = pd.DataFrame([{
                "Time":   datetime.fromtimestamp(r["timestamp"]).strftime("%H:%M:%S"),
                "User":   r["username"],
                "Action": r["action"],
                "Target": r["target_id"][:12] if r.get("target_id") else "",
                "Detail": r["detail"],
            } for r in log])
            st.dataframe(log_df, use_container_width=True, hide_index=True, height=250)
        except ImportError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — LINK TO INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_link:
    st.header("🔗 Link to Historical Incidents")

    inc_text = st.text_area("Paste incident report", height=180,
        placeholder="Paste CSB report text…")

    c1,c2,c3 = st.columns(3)
    for btn_label, fname in [
        ("TX City 2005", "csb_texas_city_2005.txt"),
        ("T2 Labs 2007",  "csb_t2_laboratories_2007.txt"),
        ("Ammonia Refrig","ammonia_refrigeration_generic.txt"),
    ]:
        if c1.button(f"Load {btn_label}"):
            p = SAMPLE_DIR / fname
            if p.exists():
                inc_text = p.read_text(encoding="utf-8")
                st.session_state["_inc_text"] = inc_text
        c1,c2,c3 = c2,c3,st.columns(3)[0]

    inc_text = inc_text or st.session_state.get("_inc_text","")

    if st.button("🔬 Extract & Seed P&ID", type="primary") and inc_text.strip():
        with st.spinner("Running NER…"):
            from src.ingestion    import ingest_text
            from src.ner_pipeline import HAZOPNERPipeline
            doc  = ingest_text(inc_text, save=False)
            pipe = HAZOPNERPipeline(use_bert=False)
            ner  = pipe.run(doc)
            pid  = pid_from_ner(ner, system_name=doc["metadata"]["source"])
            st.session_state["pid_system"] = pid
        unique = ner.get("unique_terms", {})
        st.success(
            f"✅ {len(unique.get('CHEM',[]))} chemicals · "
            f"{len(unique.get('EQUIP',[]))} equipment · "
            f"{len(pid.nodes)} P&ID nodes built"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — ADMIN / ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_admin:
    if has_role("admin"):
        from src.auth import render_user_admin
        render_user_admin()

        st.divider()
        st.subheader("🔬 NER Fine-tuning")
        st.markdown(
            "Generate a safety-domain training corpus and fine-tune BERT NER. "
            "This runs in the background and saves the model to `models/ner_finetuned/`."
        )
        col_ft1, col_ft2 = st.columns(2)
        ft_epochs     = col_ft1.slider("Training epochs", 1, 10, 3)
        ft_max_sents  = col_ft2.slider("Max training sentences", 500, 10000, 3000, step=500)
        if st.button("▶ Generate Corpus + Fine-tune NER"):
            with st.spinner("Generating training corpus…"):
                try:
                    from src.ner_trainer import generate_silver_corpus, finetune
                    corpus = generate_silver_corpus(max_sentences=ft_max_sents)
                    st.info(f"Corpus: {len(corpus)} sentences. Starting fine-tuning…")
                    out = finetune(corpus, epochs=ft_epochs)
                    st.success(f"Fine-tuned model saved to {out}")
                except Exception as e:
                    st.error(f"Fine-tuning failed: {e}  \nRequires: pip install transformers torch datasets")

        st.divider()
        st.subheader("📦 Semantic KB Management")
        kb_stats = {}
        try:
            kb = get_semantic_kb(auto_build=False)
            kb_stats = kb.stats()
        except Exception:
            kb_stats = {"status": "unavailable"}

        st.json(kb_stats)
        if st.button("🔨 Rebuild Semantic KB Index"):
            with st.spinner("Rebuilding…"):
                try:
                    kb = get_semantic_kb(auto_build=False)
                    n  = kb.build(force_rebuild=True)
                    st.success(f"Rebuilt: {n:,} chunks indexed.")
                except Exception as e:
                    st.error(f"{e}  \nRun: pip install sentence-transformers faiss-cpu")
    else:
        st.header("ℹ About HAZOP NLP v2")
        st.markdown("""
**Upgraded from v1:**
- 🔐 **Authentication** — login with roles (admin / engineer / viewer)
- 💾 **Persistent storage** — SQLite database saves all sessions and review history
- 🖼 **P&ID image parser** — GPT-4o vision extracts nodes from scanned drawings
- 🔎 **Semantic KB** — embeddings over incident reports replace hand-coded rules
- 📐 **LOPA engine** — quantified risk with CCPS initiating event frequencies and IPL PFD credits
- 🤖 **Fine-tuned NER** — BERT fine-tuner on safety-domain labels

**Run fine-tuning (admin only):**
```bash
python -m src.ner_trainer --epochs 3 --max-sents 5000
```
**Build semantic index:**
```bash
python -m src.semantic_kb --build
```
        """)
