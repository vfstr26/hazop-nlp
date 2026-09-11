# ⚗ HAZOP NLP — AI Safety Incident Analysis System

> **Automate HAZOP preparation using NLP, BERT, and LLMs to extract actionable safety data from chemical incident reports.**

Process safety engineers spend weeks reading through historical incident reports before a HAZOP study. This system reduces that to minutes — extracting chemicals, equipment, causes, consequences, and safeguards from unstructured text, then mapping them to HAZOP deviations using a built-in knowledge base.

---

## What it does

```
Incident Report (PDF / TXT / DOCX)
        │
        ▼
    NER Extraction ────► Chemicals, Equipment, Parameters,
  (BERT + Gazetteer)      Causes, Consequences, Safeguards
        │
        ▼
   HAZOP Engine ──────► Guide word × parameter deviations
   (IEC 61882)           5×5 Risk matrix, Action priorities
        │
        ▼
  LLM Enrichment ─────► Deeper causes, safeguard gaps,
  (OpenAI / Local)       Executive summary, Q&A
        │
        ▼
     Output ─────────► HTML worksheet · Excel · JSON · CSV
```

**Example output (one row of the HAZOP table):**

| Node | Equipment | Chemical | Deviation | Risk | Causes | Recommended Safeguards |
|------|-----------|----------|-----------|------|--------|------------------------|
| N001 | Reactor | Hydrogen | More temperature | 🔴 Critical (15/25) | Loss of cooling water, Exothermic runaway | Install SIS high-temp interlock, Emergency quench system |

---

## Quick Start

### 1. Clone / set up

```bash
# Windows PowerShell
cd C:\Users\chaka\Desktop\HAZOP\hazop_nlp

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure (optional — needed only for OpenAI)

```bash
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the web app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Or use the CLI

```bash
# Analyse a sample report (rule-based NER, mock LLM)
python cli.py analyse data/sample_reports/csb_texas_city_2005.txt

# Analyse with BERT NER and real OpenAI enrichment
python cli.py analyse data/sample_reports/csb_t2_laboratories_2007.txt --bert --llm openai

# Paste text interactively
python cli.py analyse --text

# List all analysed documents
python cli.py list

# Export an existing analysis to HTML
python cli.py export <doc_id> --format html

# Fetch CSB incident list from the web
python cli.py fetch-csb --pages 3
```

---

## Project Structure

```
hazop_nlp/
├── app.py                     # Streamlit web interface
├── cli.py                     # Command-line interface
├── config.py                  # Central configuration
├── requirements.txt
├── .env.example
│
├── src/
│   ├── ingestion.py           # PDF/TXT/DOCX parsing, text cleaning
│   ├── ner_pipeline.py        # BERT NER + rule-based NER
│   ├── hazop_engine.py        # HAZOP analysis, risk matrix, KB
│   ├── llm_inference.py       # OpenAI / local LLM enrichment
│   └── output_formatter.py    # JSON / HTML / Excel / CSV export
│
├── data/
│   ├── raw/                   # Place downloaded CSB PDFs here
│   ├── processed/             # Auto-saved parsed documents (JSON)
│   └── sample_reports/        # 3 ready-to-use incident reports
│       ├── csb_texas_city_2005.txt
│       ├── csb_t2_laboratories_2007.txt
│       └── ammonia_refrigeration_generic.txt
│
├── models/
│   └── llm/                   # Place local GGUF model here
│
└── outputs/                   # Generated reports (HTML, Excel, etc.)
```

---

## Data Sources

### U.S. Chemical Safety Board (CSB)
- **URL:** https://www.csb.gov/investigations/
- **Format:** PDF investigation reports (free, public domain)
- **How to use:** Download PDFs into `data/raw/`, then run `python cli.py analyse data/raw/report.pdf`
- **Notable reports:** Texas City 2005, T2 Laboratories 2007, West Fertilizer 2013, Deepwater Horizon 2010

### AIChE CCPS / EPSC
- **URL:** https://www.aiche.org/ccps and https://www.epsc.be
- **Format:** Guidelines documents, loss-of-containment case studies
- **Access:** Some free, some require AIChE membership

### ARIA (French BARPI database)
- **URL:** https://www.aria.developpement-durable.gouv.fr/
- **Format:** Searchable accident database, downloadable XML/CSV

### UK HSE
- **URL:** https://www.hse.gov.uk/comah/sragtech/
- **Format:** COMAH accident reports, free PDF

---

## NER Entity Types

| Label | Description | Examples |
|-------|-------------|---------|
| `CHEM` | Chemical / Substance | hydrogen sulfide, ammonia, propane |
| `EQUIP` | Equipment / Component | reactor, PRV, heat exchanger, pipeline |
| `PARAM` | Process Parameter | temperature, pressure, flow, level |
| `CAUSE` | Cause / Initiating Event | corrosion, valve failure, loss of cooling |
| `CONSEQ` | Consequence / Outcome | explosion, fire, fatality, toxic release |
| `SAFEGUARD` | Safeguard / Mitigation | gas detector, ESD, interlock, PRV |
| `LOC` | Location / Plant Area | refinery, storage area |
| `ORG` | Organisation | BP, CSB |
| `PERSON` | Person / Role | operator, engineer |

---

## HAZOP Guide Words (IEC 61882)

| Guide Word | Meaning |
|------------|---------|
| No / None | Complete negation of intent |
| More | Quantitative increase |
| Less | Quantitative decrease |
| As Well As | Additional activity |
| Part Of | Qualitative decrease |
| Reverse | Logical opposite |
| Other Than | Complete substitution |
| Early / Late / Before / After | Timing deviations |

---

## LLM Configuration

### Option A — OpenAI API (recommended for best quality)

1. Get an API key at https://platform.openai.com/api-keys
2. Add to `.env`: `OPENAI_API_KEY=sk-...`
3. Select "openai" in the Streamlit sidebar or use `--llm openai` in CLI
4. Default model: `gpt-4o-mini` (~$0.002 per analysis, adjustable in `.env`)

### Option B — Local GGUF model (no API key, runs offline)

1. Download a GGUF model:
   - **Mistral 7B Instruct Q4:** https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
   - **Llama-3 8B Instruct Q4:** https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF
2. Place `.gguf` file in `models/llm/`
3. Set `LOCAL_MODEL_PATH` in `.env`
4. Select "local (GGUF)" in Streamlit or use `--llm local` in CLI
5. Install with GPU support: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`

### Option C — Mock mode (no API key, no model)

Default behaviour when no API key is configured. Returns placeholder text so the full pipeline still runs and produces outputs.

---

## Risk Matrix

| | L1 Rare | L2 Unlikely | L3 Possible | L4 Likely | L5 Almost Certain |
|---|---------|-------------|-------------|-----------|-------------------|
| **S5 Catastrophic** | 🟡 Med | 🟠 High | 🔴 Crit | 🔴 Crit | 🔴 Crit |
| **S4 Major** | 🟢 Low | 🟡 Med | 🟠 High | 🔴 Crit | 🔴 Crit |
| **S3 Moderate** | 🟢 Low | 🟢 Low | 🟡 Med | 🟠 High | 🔴 Crit |
| **S2 Minor** | 🟢 Low | 🟢 Low | 🟢 Low | 🟡 Med | 🟠 High |
| **S1 Negligible** | 🟢 Low | 🟢 Low | 🟢 Low | 🟢 Low | 🟡 Med |

---

## BERT NER Model

The system uses [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) from HuggingFace, a BERT model fine-tuned on CoNLL-2003 NER.

- **First run:** Downloads ~430 MB model automatically to HuggingFace cache
- **Entities extracted:** PER, ORG, LOC, MISC
- **Combined with rule-based NER** for domain-specific terms (chemicals, equipment, safeguards)

To fine-tune on your own labelled incident data:

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
# See HuggingFace token classification fine-tuning guide:
# https://huggingface.co/docs/transformers/tasks/token_classification
```

---

## Extending the Knowledge Base

The HAZOP deviation knowledge base is in `src/hazop_engine.py` under `DEVIATION_KB`.
Add entries following this pattern:

```python
("guide_word_lowercase", "parameter_lowercase"): {
    "causes":       ["Cause 1", "Cause 2"],
    "consequences": ["Consequence 1"],
    "safeguards":   ["Existing safeguard 1"],
    "recommended":  ["Recommendation 1", "Recommendation 2"],
    "severity":     4,    # 1–5
    "likelihood":   3,    # 1–5
    "ref":          "Source reference",
},
```

---

## Disclaimer

> This tool is intended to **assist** qualified process safety engineers during HAZOP preparation.
> It does **not** replace a formal HAZOP study conducted by a multidisciplinary team under the supervision of a certified HAZOP facilitator.
> All outputs must be reviewed and validated by competent engineers.
> Do not use in safety-critical decisions without expert review.

---

## References

- IEC 61882:2016 — Hazard and Operability Studies (HAZOP) — Application Guide
- CCPS *Guidelines for Hazard Evaluation Procedures*, 3rd Edition
- CCPS *Layer of Protection Analysis: Simplified Process Risk Assessment*
- U.S. Chemical Safety Board — https://www.csb.gov
- AIChE Center for Chemical Process Safety — https://www.aiche.org/ccps
- Kletz, T. (2003). *What Went Wrong? — Case Histories of Process Plant Disasters*, 5th Ed.
- Lees, F. (2012). *Loss Prevention in the Process Industries*, 4th Ed.
#   h a z o p - n l p  
 