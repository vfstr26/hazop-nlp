"""
app.py — Streamlit web interface for the HAZOP NLP Safety Analysis System.

Run:
    streamlit run app.py

Tabs:
  1. Analyse    — paste text or upload a file, run the full pipeline
  2. Results    — view HAZOP table, filter by risk/priority
  3. Q&A        — ask the LLM questions about the incident
  4. Export     — download JSON / HTML / Excel / CSV
  5. About      — methodology and data sources
"""

import sys
import json
import tempfile
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from src.ingestion        import ingest_text, ingest_file
from src.ner_pipeline     import HAZOPNERPipeline, format_ner_report
from src.hazop_engine     import HAZOPEngine, summarise_hazop, rows_to_dicts
from src.llm_inference    import get_llm_client, generate_summary, answer_question, enrich_all_rows
from src.output_formatter import to_html, to_json, to_csv, to_excel, export_all
from config               import OUTPUT_DIR, SAMPLE_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAZOP NLP Analyser",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour constants (used in st.markdown tables) ─────────────────────────────
RISK_COLOURS = {
    "Critical": "#C0392B",
    "High":     "#E67E22",
    "Medium":   "#F1C40F",
    "Low":      "#27AE60",
}
RISK_TEXT_COLOURS = {
    "Critical": "white", "High": "white", "Medium": "#2C3E50", "Low": "white"
}


# ══════════════════════════════════════════════════════════════════════════════
# Session State Initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "document":    None,
        "ner_result":  None,
        "hazop_rows":  None,
        "summary":     None,
        "llm_summary": None,
        "doc_id":      "analysis",
        "context_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _risk_badge(level: str) -> str:
    bg = RISK_COLOURS.get(level, "#7F8C8D")
    fg = RISK_TEXT_COLOURS.get(level, "white")
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 9px;'
        f'border-radius:4px;font-weight:700;font-size:0.8em">{level}</span>'
    )


def _priority_badge(pri: str) -> str:
    colours = {"Immediate": "#C0392B", "Short-term": "#E67E22", "Long-term": "#27AE60"}
    bg = colours.get(pri, "#95A5A6")
    return (
        f'<span style="background:{bg};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78em">{pri}</span>'
    )


def _safe_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val] if val else []
    return []


def _risk_score(row: dict) -> int:
    risk = row.get("risk", {})
    if isinstance(risk, dict):
        return risk.get("risk_score", 0)
    return 0


def _risk_level(row: dict) -> str:
    risk = row.get("risk", {})
    if isinstance(risk, dict):
        return risk.get("risk_level", "Low")
    return "Low"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Chemical_Engineering_symbol.svg/120px-Chemical_Engineering_symbol.svg.png",
             width=60, caption="")
    st.title("⚗ HAZOP NLP")
    st.caption("AI-powered Safety Analysis")
    st.divider()

    st.subheader("⚙ Settings")

    use_bert = st.toggle(
        "Use BERT NER model",
        value=False,
        help="Enable HuggingFace BERT NER for improved entity extraction. "
             "Requires transformers package and ~500 MB model download.",
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        ["mock (no API key)", "openai", "local (GGUF)"],
        help="'mock' runs without an API key. 'openai' needs OPENAI_API_KEY in .env.",
    )

    enrich_llm = st.toggle(
        "LLM enrichment",
        value=True,
        help="Use LLM to enrich top high-risk HAZOP rows with deeper analysis.",
    )

    max_enrich = st.slider("Max rows to enrich", 1, 30, 10)
    st.divider()

    st.subheader("📂 Load Sample")
    sample_files = list(SAMPLE_DIR.glob("*.txt")) + list(SAMPLE_DIR.glob("*.pdf"))
    sample_names = ["— select —"] + [f.name for f in sample_files]
    selected_sample = st.selectbox("Sample reports", sample_names)

    if st.button("Load Sample", use_container_width=True) and selected_sample != "— select —":
        sample_path = SAMPLE_DIR / selected_sample
        st.session_state["context_text"] = sample_path.read_text(encoding="utf-8", errors="replace")
        st.toast(f"Loaded: {selected_sample}", icon="📄")

    st.divider()
    st.caption("Data sources: U.S. Chemical Safety Board (csb.gov) · AIChE EPSC · CCPS")


# ══════════════════════════════════════════════════════════════════════════════
# Main Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_analyse, tab_results, tab_qa, tab_export, tab_about = st.tabs(
    ["🔬 Analyse", "📋 Results", "💬 Q&A", "📥 Export", "ℹ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

with tab_analyse:
    st.header("🔬 Incident Analysis")
    st.markdown(
        "Paste an incident report, upload a file, or load a sample. "
        "The pipeline runs: **Ingestion → NER → HAZOP Engine → (LLM Enrichment)**"
    )

    col_input, col_upload = st.columns([3, 1])

    with col_upload:
        st.subheader("Upload file")
        uploaded = st.file_uploader(
            "PDF / TXT / DOCX",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            doc = ingest_file(tmp_path, save=False)
            st.session_state["context_text"] = doc["full_text"][:8000]
            st.toast(f"Uploaded: {uploaded.name}", icon="📎")
            tmp_path.unlink(missing_ok=True)

    with col_input:
        st.subheader("Incident report text")
        text_input = st.text_area(
            "Paste text here",
            value=st.session_state.get("context_text", ""),
            height=260,
            placeholder=(
                "Paste a chemical incident report here…\n\n"
                "Example: 'A hydrogen sulfide leak from a corroded pipeline caused "
                "an explosion resulting in two fatalities…'"
            ),
            label_visibility="collapsed",
        )
        st.session_state["context_text"] = text_input

    if st.button("🚀 Run HAZOP Analysis", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Please enter or upload incident text first.")
        else:
            with st.spinner("Step 1/3 — Running NER extraction..."):
                doc = ingest_text(text_input, source_name="user_input", save=False)
                pipe = HAZOPNERPipeline(use_bert=use_bert)
                ner_result = pipe.run(doc)
                st.session_state["document"]   = doc
                st.session_state["ner_result"] = ner_result
                st.session_state["doc_id"]     = doc["metadata"]["doc_id"]

            with st.spinner("Step 2/3 — Running HAZOP analysis engine..."):
                engine = HAZOPEngine()
                rows   = engine.run(ner_result)
                dicts  = rows_to_dicts(rows)
                summary = summarise_hazop(rows)
                st.session_state["hazop_rows"] = dicts
                st.session_state["summary"]    = summary

            if enrich_llm and dicts:
                provider_map = {
                    "mock (no API key)": "mock",
                    "openai": "openai",
                    "local (GGUF)": "local",
                }
                with st.spinner(f"Step 3/3 — LLM enrichment ({llm_provider})..."):
                    client = get_llm_client(provider_map.get(llm_provider, "mock"))
                    dicts  = enrich_all_rows(dicts, text_input, client, max_rows=max_enrich)
                    st.session_state["hazop_rows"] = dicts
                    # Generate LLM narrative summary
                    st.session_state["llm_summary"] = generate_summary(
                        dicts, summary, text_input, client
                    )

            st.success(
                f"✅ Analysis complete — **{summary['total_rows']} HAZOP rows** | "
                f"🔴 {summary['risk_breakdown']['Critical']} Critical  "
                f"🟠 {summary['risk_breakdown']['High']} High  "
                f"🟡 {summary['risk_breakdown']['Medium']} Medium  "
                f"🟢 {summary['risk_breakdown']['Low']} Low"
            )
            st.info("View results in the **Results** tab →")

    # ── NER Preview ───────────────────────────────────────────────────────────
    if st.session_state["ner_result"]:
        with st.expander("🔎 NER Extraction Preview", expanded=False):
            ner = st.session_state["ner_result"]
            cols = st.columns(3)
            label_groups = {
                "Chemicals":  "CHEM",
                "Equipment":  "EQUIP",
                "Parameters": "PARAM",
                "Causes":     "CAUSE",
                "Consequences": "CONSEQ",
                "Safeguards": "SAFEGUARD",
            }
            for i, (name, label) in enumerate(label_groups.items()):
                with cols[i % 3]:
                    terms = ner.get("unique_terms", {}).get(label, [])
                    st.markdown(f"**{name}** ({len(terms)})")
                    if terms:
                        for t in sorted(terms)[:8]:
                            st.markdown(f"- `{t}`")
                    else:
                        st.caption("None found")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.header("📋 HAZOP Analysis Results")

    if not st.session_state["hazop_rows"]:
        st.info("Run an analysis in the **Analyse** tab first.")
    else:
        rows   = st.session_state["hazop_rows"]
        summary = st.session_state["summary"]

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Rows",    summary["total_rows"])
        k2.metric("🔴 Critical",   summary["risk_breakdown"].get("Critical", 0))
        k3.metric("🟠 High",       summary["risk_breakdown"].get("High", 0))
        k4.metric("🟡 Medium",     summary["risk_breakdown"].get("Medium", 0))
        k5.metric("🟢 Low",        summary["risk_breakdown"].get("Low", 0))
        k6.metric("⚠ Immediate",   summary["priority_breakdown"].get("Immediate", 0))

        st.divider()

        # ── LLM Summary ───────────────────────────────────────────────────────
        if st.session_state.get("llm_summary"):
            with st.expander("🧠 LLM Executive Summary", expanded=True):
                st.markdown(st.session_state["llm_summary"])

        # ── Filters ───────────────────────────────────────────────────────────
        st.subheader("Filters")
        fc1, fc2, fc3 = st.columns(3)
        filter_risk = fc1.multiselect(
            "Risk Level",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
        )
        filter_priority = fc2.multiselect(
            "Priority",
            ["Immediate", "Short-term", "Long-term"],
            default=["Immediate", "Short-term", "Long-term"],
        )
        all_params = sorted({r.get("parameter", "") for r in rows})
        filter_param = fc3.multiselect("Parameter", all_params, default=all_params)

        filtered = [
            r for r in rows
            if _risk_level(r)          in filter_risk
            and r.get("action_priority") in filter_priority
            and r.get("parameter", "")  in filter_param
        ]

        st.caption(f"Showing {len(filtered)} of {len(rows)} rows")
        st.divider()

        # ── Table rows ────────────────────────────────────────────────────────
        for row in sorted(filtered, key=_risk_score, reverse=True):
            risk  = row.get("risk", {})
            level = risk.get("risk_level", "Low") if isinstance(risk, dict) else "Low"
            score = risk.get("risk_score", 0)      if isinstance(risk, dict) else 0

            with st.expander(
                f"{_risk_badge(level)} &nbsp; **{row.get('deviation','')}** — "
                f"{row.get('equipment','')} / {row.get('chemical','')} "
                f"(Score: {score}/25) &nbsp; {_priority_badge(row.get('action_priority',''))}",
                expanded=(level in ("Critical", "High")),
            ):
                c1, c2, c3 = st.columns([1, 1, 1])

                with c1:
                    st.markdown("**📌 Node**")
                    st.code(
                        f"Node:      {row.get('node_id','')}\n"
                        f"Equipment: {row.get('equipment','')}\n"
                        f"Chemical:  {row.get('chemical','')}\n"
                        f"Parameter: {row.get('parameter','')}\n"
                        f"Guide Word:{row.get('guide_word','')}"
                    )
                    st.markdown("**📖 Historical Reference**")
                    st.caption(row.get("historical_ref", "N/A"))

                with c2:
                    st.markdown("**⚡ Causes**")
                    for cause in _safe_list(row.get("causes", [])):
                        st.markdown(f"- {cause}")
                    st.markdown("**💥 Consequences**")
                    for cons in _safe_list(row.get("consequences", [])):
                        st.markdown(f"- {cons}")

                with c3:
                    st.markdown("**🛡 Existing Safeguards**")
                    for sg in _safe_list(row.get("safeguards_existing", [])):
                        st.markdown(f"- {sg}")
                    st.markdown("**✅ Recommended Safeguards**")
                    for rec in _safe_list(row.get("safeguards_recommended", [])):
                        st.markdown(f"- {rec}")

                st.markdown("**🔧 Actions**")
                for act in _safe_list(row.get("actions", [])):
                    st.markdown(f"→ {act}")
                if row.get("notes"):
                    st.caption(f"Notes: {row['notes']}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Q&A
# ══════════════════════════════════════════════════════════════════════════════

with tab_qa:
    st.header("💬 Incident Q&A")
    st.markdown(
        "Ask the LLM specific questions about the incident report. "
        "Answers are grounded in the source text."
    )

    if not st.session_state["context_text"]:
        st.info("Load or paste incident text in the **Analyse** tab first.")
    else:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What was the root cause of the explosion?",
        )

        preset_qs = [
            "What was the root cause of the incident?",
            "What chemicals were involved and what are their hazards?",
            "What safeguards failed or were absent?",
            "What recommendations were made to prevent recurrence?",
            "What were the consequences in terms of casualties and damage?",
        ]
        selected_preset = st.selectbox("Or choose a preset question", ["— custom —"] + preset_qs)
        if selected_preset != "— custom —":
            question = selected_preset

        if st.button("Ask", type="primary") and question:
            provider_map = {
                "mock (no API key)": "mock",
                "openai": "openai",
                "local (GGUF)": "local",
            }
            with st.spinner("Thinking..."):
                client = get_llm_client(provider_map.get(llm_provider, "mock"))
                answer = answer_question(
                    question,
                    st.session_state["context_text"],
                    client,
                )
            st.markdown("### Answer")
            st.markdown(answer)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.header("📥 Export Results")

    if not st.session_state["hazop_rows"]:
        st.info("Run an analysis first.")
    else:
        rows    = st.session_state["hazop_rows"]
        summary = st.session_state["summary"]
        meta    = st.session_state["document"]["metadata"] if st.session_state["document"] else {}
        ner     = st.session_state["ner_result"]
        doc_id  = st.session_state["doc_id"]

        st.markdown("Download your analysis in multiple formats.")
        ec1, ec2, ec3, ec4 = st.columns(4)

        # JSON
        with ec1:
            json_str = to_json(rows, summary, meta)
            st.download_button(
                "⬇ JSON",
                data=json_str,
                file_name=f"{doc_id}_hazop.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Machine-readable full dataset")

        # HTML
        with ec2:
            html_str = to_html(rows, summary, meta)
            st.download_button(
                "⬇ HTML Report",
                data=html_str,
                file_name=f"{doc_id}_hazop.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Print-ready worksheet")

        # CSV
        with ec3:
            csv_str = to_csv(rows)
            st.download_button(
                "⬇ CSV",
                data=csv_str,
                file_name=f"{doc_id}_hazop.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Flat table for Excel / BI tools")

        # Excel
        with ec4:
            try:
                excel_bytes = to_excel(rows, summary, meta, ner)
                st.download_button(
                    "⬇ Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"{doc_id}_hazop.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption("Multi-sheet workbook with colour-coded risk")
            except RuntimeError as e:
                st.warning(f"Excel unavailable: {e}")

        st.divider()
        st.subheader("Preview — HTML Report")
        html_preview = to_html(rows, summary, meta)
        st.components.v1.html(html_preview, height=600, scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.header("ℹ About HAZOP NLP")
    st.markdown("""
## What is this tool?

This system uses **Natural Language Processing (NLP)** to automate the first-pass
analysis of chemical process safety, dramatically reducing the time engineers spend
reviewing historical incident reports before a HAZOP study.

---

## Pipeline Architecture

```
Incident Report (PDF / TXT / DOCX / paste)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  1. INGESTION                           │
  │  pdfplumber / PyMuPDF / python-docx     │
  │  Text cleaning, sentence segmentation  │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────────┐
  │  2. NER (Named Entity Recognition)      │
  │  • BERT layer (dslim/bert-base-NER)     │
  │  • Rule/Gazetteer layer (120+ chemicals,│
  │    30 equipment types, 25 causes, etc.) │
  │  Entities: CHEM, EQUIP, PARAM, CAUSE,  │
  │            CONSEQ, SAFEGUARD           │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────────┐
  │  3. HAZOP ENGINE (IEC 61882)            │
  │  • Node identification                  │
  │  • Guide word application               │
  │  • Deviation KB lookup (14 deviations) │
  │  • 5×5 Risk matrix                      │
  │  • Action generation                    │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────────┐
  │  4. LLM ENRICHMENT (optional)           │
  │  • OpenAI API / Local GGUF model        │
  │  • Enrich causes, safeguards, actions   │
  │  • Executive summary generation         │
  │  • Incident Q&A                         │
  └───────────────┬─────────────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────────────┐
  │  5. OUTPUT                              │
  │  JSON  │  HTML  │  Excel  │  CSV        │
  └─────────────────────────────────────────┘
```

---

## Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| U.S. Chemical Safety Board (CSB) | Public incident investigation reports | [csb.gov/investigations](https://www.csb.gov/investigations/) |
| AIChE CCPS | Guidelines for Hazard Evaluation Procedures | [aiche.org/ccps](https://www.aiche.org/ccps) |
| EPSC | European Process Safety Centre loss-of-containment data | [epsc.be](https://www.epsc.be) |
| IEC 61882 | HAZOP study standard | ISO/IEC |

---

## HAZOP Guide Words (IEC 61882)

| Guide Word | Meaning |
|-----------|---------|
| No / None | Complete negation of the design intent |
| More | Quantitative increase |
| Less | Quantitative decrease |
| As Well As | Qualitative increase / additional activity |
| Part Of | Qualitative decrease |
| Reverse | Logical opposite |
| Other Than | Complete substitution |
| Early | Relative to clock time |
| Late | Relative to clock time |
| Before | Relating to order or sequence |
| After | Relating to order or sequence |

---

## Disclaimer

> This tool is intended to **assist** qualified process safety engineers.
> It does **not** replace a formal HAZOP study conducted by a multidisciplinary team
> under the supervision of a certified facilitator.
> All outputs must be reviewed and validated by competent engineers before use
> in safety-critical decisions.

**References:** IEC 61882:2016 · CCPS *Guidelines for Hazard Evaluation Procedures* (3rd Ed.)
· *Layer of Protection Analysis* (CCPS) · CSB Accident Investigation Reports
    """)
