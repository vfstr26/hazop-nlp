"""
smoke_test.py — Verifies the full pipeline runs end-to-end without errors.
Run: python smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion        import ingest_text
from src.ner_pipeline     import HAZOPNERPipeline
from src.hazop_engine     import HAZOPEngine, summarise_hazop, rows_to_dicts
from src.llm_inference    import MockLLMClient, enrich_all_rows
from src.output_formatter import to_html, to_json, to_csv

SAMPLE = (
    "A hydrogen sulfide leak from a corroded pipeline at the refinery caused "
    "an explosion resulting in two fatalities. The pressure relief valve had failed "
    "to open due to corrosion. The reactor temperature rose rapidly following loss "
    "of cooling water. Recommended safeguards: install gas detectors and upgrade PRVs, "
    "implement an emergency shutdown interlock on cooling water loss."
)

errors = []

# 1 — Ingest
try:
    doc = ingest_text(SAMPLE, source_name="smoke_test", save=False)
    assert len(doc["sentences"]) > 0
    print(f"[PASS] 1/5 Ingest      — {len(doc['sentences'])} sentences, {doc['metadata']['char_count']} chars")
except Exception as e:
    errors.append(f"Ingest FAILED: {e}")
    print(f"[FAIL] 1/5 Ingest      — {e}")

# 2 — NER
try:
    pipe = HAZOPNERPipeline(use_bert=False)
    ner  = pipe.run(doc)
    assert ner["total"] > 0
    print(f"[PASS] 2/5 NER         — {ner['total']} entities, labels: {list(ner['summary'].keys())}")
except Exception as e:
    errors.append(f"NER FAILED: {e}")
    print(f"[FAIL] 2/5 NER         — {e}")

# 3 — HAZOP Engine
try:
    engine = HAZOPEngine()
    rows   = engine.run(ner)
    dicts  = rows_to_dicts(rows)
    summ   = summarise_hazop(rows)
    assert summ["total_rows"] > 0
    print(f"[PASS] 3/5 HAZOP Engine — {summ['total_rows']} rows | "
          f"Critical={summ['risk_breakdown']['Critical']} "
          f"High={summ['risk_breakdown']['High']} "
          f"Medium={summ['risk_breakdown']['Medium']} "
          f"Low={summ['risk_breakdown']['Low']}")
except Exception as e:
    errors.append(f"HAZOP Engine FAILED: {e}")
    print(f"[FAIL] 3/5 HAZOP Engine — {e}")

# 4 — LLM Enrichment (mock)
try:
    client = MockLLMClient()
    dicts  = enrich_all_rows(dicts, SAMPLE, client, max_rows=3, delay=0)
    n_enriched = sum(1 for r in dicts if r.get("llm_enriched"))
    print(f"[PASS] 4/5 LLM (mock)  — {n_enriched} rows enriched")
except Exception as e:
    errors.append(f"LLM FAILED: {e}")
    print(f"[FAIL] 4/5 LLM         — {e}")

# 5 — Output formatters
try:
    meta = doc["metadata"]
    html = to_html(dicts, summ, meta)
    jsn  = to_json(dicts, summ, meta)
    csv_ = to_csv(dicts)
    assert "<table>" in html
    assert '"hazop_table"' in jsn
    assert "node_id" in csv_
    print(f"[PASS] 5/5 Output      — HTML={len(html):,} chars  JSON={len(jsn):,} chars  CSV={len(csv_.splitlines())} lines")
except Exception as e:
    errors.append(f"Output FAILED: {e}")
    print(f"[FAIL] 5/5 Output      — {e}")

# Summary
print()
if errors:
    print(f"=== {len(errors)} TEST(S) FAILED ===")
    for err in errors:
        print(f"  • {err}")
    sys.exit(1)
else:
    print("=== ALL TESTS PASSED ===")
    print()
    print("Next steps:")
    print("  streamlit run app.py")
    print("  python cli.py analyse data/sample_reports/csb_texas_city_2005.txt")
