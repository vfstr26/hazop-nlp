"""
cli.py — Command-line interface for the HAZOP NLP system.

Usage examples:
  # Analyse a file
  python cli.py analyse report.pdf

  # Analyse pasted text (interactive)
  python cli.py analyse --text

  # Analyse with BERT NER and OpenAI enrichment
  python cli.py analyse report.txt --bert --llm openai

  # Download CSB incident list
  python cli.py fetch-csb --pages 2

  # List processed documents
  python cli.py list

  # Export existing analysis
  python cli.py export <doc_id> --format html
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import OUTPUT_DIR, PROCESSED_DIR


def cmd_analyse(args):
    from src.ingestion     import ingest_file, ingest_text
    from src.ner_pipeline  import HAZOPNERPipeline, format_ner_report
    from src.hazop_engine  import HAZOPEngine, summarise_hazop, rows_to_dicts
    from src.llm_inference import get_llm_client, enrich_all_rows, generate_summary
    from src.output_formatter import export_all

    # ── Get text ──────────────────────────────────────────────────────────────
    if args.text:
        print("Paste incident text below. Enter a blank line followed by END to finish:\n")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        raw_text = "\n".join(lines)
        doc = ingest_text(raw_text, source_name="cli_input", save=False)
    else:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: file not found: {filepath}")
            sys.exit(1)
        print(f"Ingesting: {filepath.name}")
        doc = ingest_file(filepath, save=True)

    print(f"  Document ID : {doc['metadata']['doc_id']}")
    print(f"  Characters  : {doc['metadata']['char_count']}")
    print(f"  Sentences   : {len(doc['sentences'])}")

    # ── NER ───────────────────────────────────────────────────────────────────
    print(f"\nRunning NER (BERT={'yes' if args.bert else 'no (rule-based)'})...")
    pipe       = HAZOPNERPipeline(use_bert=args.bert)
    ner_result = pipe.run(doc)
    print(format_ner_report(ner_result))

    # ── HAZOP engine ──────────────────────────────────────────────────────────
    print("\nRunning HAZOP analysis engine...")
    engine  = HAZOPEngine()
    rows    = engine.run(ner_result)
    dicts   = rows_to_dicts(rows)
    summary = summarise_hazop(rows)

    print(f"\n{'='*50}")
    print("HAZOP SUMMARY")
    print(f"{'='*50}")
    print(f"  Total rows       : {summary['total_rows']}")
    print(f"  Critical         : {summary['risk_breakdown']['Critical']}")
    print(f"  High             : {summary['risk_breakdown']['High']}")
    print(f"  Medium           : {summary['risk_breakdown']['Medium']}")
    print(f"  Low              : {summary['risk_breakdown']['Low']}")
    print(f"  Immediate actions: {summary['priority_breakdown']['Immediate']}")

    # ── LLM enrichment ────────────────────────────────────────────────────────
    if args.llm and args.llm != "none":
        print(f"\nEnriching with LLM ({args.llm})...")
        client = get_llm_client(args.llm)
        dicts  = enrich_all_rows(dicts, doc["full_text"], client, max_rows=args.max_enrich)
        llm_sum = generate_summary(dicts, summary, doc["full_text"], client)
        print("\n── LLM Executive Summary ──")
        print(llm_sum)

    # ── Export ────────────────────────────────────────────────────────────────
    doc_id = doc["metadata"]["doc_id"]
    fmt    = args.format or "all"
    meta   = doc["metadata"]

    if fmt == "all":
        from src.output_formatter import export_all
        paths = export_all(dicts, summary, meta, ner_result, doc_id=doc_id)
        print("\n── Exported files ──")
        for f, p in paths.items():
            print(f"  {f:6s} → {p}")
    else:
        from src.output_formatter import to_html, to_json, to_csv, to_excel
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            p = OUTPUT_DIR / f"{doc_id}_hazop.html"
            to_html(dicts, summary, meta, p)
            print(f"HTML → {p}")
        elif fmt == "json":
            p = OUTPUT_DIR / f"{doc_id}_hazop.json"
            to_json(dicts, summary, meta, p)
            print(f"JSON → {p}")
        elif fmt == "csv":
            p = OUTPUT_DIR / f"{doc_id}_hazop.csv"
            to_csv(dicts, p)
            print(f"CSV → {p}")
        elif fmt == "excel":
            p = OUTPUT_DIR / f"{doc_id}_hazop.xlsx"
            to_excel(dicts, summary, meta, ner_result, p)
            print(f"Excel → {p}")


def cmd_fetch_csb(args):
    from src.ingestion import fetch_csb_incident_list, download_csb_report
    print(f"Fetching CSB incident list ({args.pages} pages)...")
    incidents = fetch_csb_incident_list(max_pages=args.pages)
    print(f"Found {len(incidents)} incidents:\n")
    for i, inc in enumerate(incidents[:20], 1):
        print(f"  {i:2d}. {inc['title']} ({inc['date']})")
        print(f"       {inc['url']}")
    if args.download and incidents:
        print(f"\nDownloading first {args.download} PDF reports...")
        for inc in incidents[:args.download]:
            if inc["url"].endswith(".pdf"):
                download_csb_report(inc["url"])


def cmd_list(args):
    files = list(PROCESSED_DIR.glob("*.json"))
    if not files:
        print("No processed documents found in data/processed/")
        return
    print(f"{'Doc ID':<12} {'Source':<35} {'Date':<20} {'Chars':>8}")
    print("-" * 78)
    for f in sorted(files):
        if f.stem.endswith("_ner"):
            continue
        with open(f, encoding="utf-8") as fh:
            meta = json.load(fh).get("metadata", {})
        print(f"  {meta.get('doc_id',''):<10} {meta.get('source',''):<35} "
              f"{meta.get('date',''):<20} {meta.get('char_count',0):>8,}")


def cmd_export(args):
    from src.hazop_engine     import load_hazop_rows
    from src.output_formatter import export_all, to_html, to_json, to_csv, to_excel
    from src.ingestion        import load_processed

    doc_id = args.doc_id
    print(f"Loading analysis for doc_id: {doc_id}")

    try:
        rows = load_hazop_rows(doc_id)
    except FileNotFoundError:
        print(f"No HAZOP analysis found for {doc_id}. Run 'analyse' first.")
        sys.exit(1)

    try:
        doc  = load_processed(doc_id)
        meta = doc.get("metadata", {"doc_id": doc_id, "source": doc_id})
    except FileNotFoundError:
        meta = {"doc_id": doc_id, "source": doc_id, "date": "Unknown", "chemicals": []}

    summary = {
        "total_rows": len(rows),
        "unique_nodes": len({r.get("node_id") for r in rows}),
        "unique_deviations": len({r.get("deviation") for r in rows}),
        "risk_breakdown": {k: sum(1 for r in rows if isinstance(r.get("risk"), dict) and r["risk"].get("risk_level") == k)
                           for k in ["Critical", "High", "Medium", "Low"]},
        "priority_breakdown": {k: sum(1 for r in rows if r.get("action_priority") == k)
                                for k in ["Immediate", "Short-term", "Long-term"]},
    }

    fmt = args.format or "all"
    if fmt == "all":
        paths = export_all(rows, summary, meta, doc_id=doc_id)
        for f, p in paths.items():
            print(f"  {f:6s} → {p}")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            p = OUTPUT_DIR / f"{doc_id}_hazop.html"
            to_html(rows, summary, meta, p); print(f"→ {p}")
        elif fmt == "json":
            p = OUTPUT_DIR / f"{doc_id}_hazop.json"
            to_json(rows, summary, meta, p); print(f"→ {p}")
        elif fmt == "csv":
            p = OUTPUT_DIR / f"{doc_id}_hazop.csv"
            to_csv(rows, p); print(f"→ {p}")


def main():
    parser = argparse.ArgumentParser(
        prog="hazop-nlp",
        description="HAZOP NLP — AI-powered Safety Incident Analysis",
    )
    subs = parser.add_subparsers(dest="command")

    # ── analyse ───────────────────────────────────────────────────────────────
    p_analyse = subs.add_parser("analyse", help="Analyse an incident report")
    p_analyse.add_argument("file", nargs="?", default="", help="Path to PDF/TXT/DOCX file")
    p_analyse.add_argument("--text", action="store_true", help="Read text interactively from stdin")
    p_analyse.add_argument("--bert", action="store_true", help="Enable BERT NER model")
    p_analyse.add_argument("--llm", default="mock",
                           choices=["none", "mock", "openai", "local"],
                           help="LLM provider for enrichment")
    p_analyse.add_argument("--max-enrich", type=int, default=10,
                           help="Max HAZOP rows to enrich with LLM (default: 10)")
    p_analyse.add_argument("--format", default="all",
                           choices=["all", "html", "json", "csv", "excel"],
                           help="Output format (default: all)")

    # ── fetch-csb ─────────────────────────────────────────────────────────────
    p_csb = subs.add_parser("fetch-csb", help="Fetch CSB incident list from the web")
    p_csb.add_argument("--pages", type=int, default=2, help="Number of CSB list pages to fetch")
    p_csb.add_argument("--download", type=int, default=0,
                       help="Number of PDF reports to download (0=none)")

    # ── list ──────────────────────────────────────────────────────────────────
    subs.add_parser("list", help="List all processed documents")

    # ── export ────────────────────────────────────────────────────────────────
    p_export = subs.add_parser("export", help="Export a previously analysed document")
    p_export.add_argument("doc_id", help="Document ID from 'list' command")
    p_export.add_argument("--format", default="all",
                          choices=["all", "html", "json", "csv", "excel"])

    args = parser.parse_args()

    if args.command == "analyse":
        cmd_analyse(args)
    elif args.command == "fetch-csb":
        cmd_fetch_csb(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
