"""Merge the model's summaries with call metadata into local deliverables.

Reads results.jsonl (from predict.py on the pod) and writes, per call, a JSON
holding the new summary next to the original Supabase one for comparison, plus
a flat CSV index.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")
INDEX = r"C:\Projects\meeting-summarizer\data\cleaned_index.csv"
ORIGINALS = r"C:\Projects\meeting-summarizer\_supabase_summaries.json"
OUT_DIR = r"C:\Projects\meeting-summarizer\output_v2"


def main():
    results = [json.loads(l) for l in open(RESULTS, encoding="utf-8") if l.strip()]
    with open(INDEX, encoding="utf-8-sig") as f:
        meta = {r["call_id"]: r for r in csv.DictReader(f)}
    originals = {r["call_id"]: r.get("summmary") for r in json.load(open(ORIGINALS, encoding="utf-8"))}

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for r in results:
        cid = r["call_id"]
        m = meta.get(cid, {})
        rec = {
            "call_id": cid,
            "title": r.get("title") or m.get("title", ""),
            "call_date": (m.get("call_timestamp", "") or "")[:10],
            "duration_seconds": int(m.get("duration_seconds") or 0),
            "direction": m.get("direction", ""),
            "deal_outcome": m.get("deal_outcome", ""),
            "summary": r["summary"],
            "original_summary": originals.get(cid, ""),
        }
        with open(os.path.join(OUT_DIR, f"{cid}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        rows.append({
            "call_id": cid,
            "title": rec["title"],
            "call_date": rec["call_date"],
            "chars": len(rec["summary"]),
            "summary": rec["summary"],
        })

    csv_path = os.path.join(OUT_DIR, "_all_summaries.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["call_id", "title", "call_date", "chars", "summary"])
        w.writeheader()
        w.writerows(rows)

    lens = sorted(r["chars"] for r in rows)
    n = len(lens)
    print(f"wrote {len(rows)} JSONs + CSV -> {OUT_DIR}")
    if n:
        print("summary chars: min=%d median=%d max=%d" % (lens[0], lens[n // 2], lens[-1]))
    empties = [r["call_id"] for r in rows if r["chars"] < 50]
    if empties:
        print(f"WARNING: {len(empties)} near-empty summaries: {empties}")


if __name__ == "__main__":
    main()
