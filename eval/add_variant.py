"""Merge a model's inference output into the frozen eval set as a new variant.

Reads results.jsonl (one {call_id, summary} per line) from predict.py and adds it
to eval_set.jsonl under a new variant key, so score.py can score it against the
same reference summaries without hand-editing the frozen set.

Run once per new model, then score with the default score.py.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="predict.py results.jsonl")
    ap.add_argument("--variant", required=True, help="new variant key (e.g. qwen4b)")
    ap.add_argument("--eval-set", default=os.path.join(HERE, "eval_set.jsonl"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or args.eval_set
    summaries = {}
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            summaries[r["call_id"]] = r.get("summary", "")

    rows = [json.loads(l) for l in open(args.eval_set, encoding="utf-8") if l.strip()]
    seen = 0
    for row in rows:
        if row["call_id"] in summaries:
            row["variants"][args.variant] = summaries[row["call_id"]]
            seen += 1

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"added variant '{args.variant}' to {seen}/{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
