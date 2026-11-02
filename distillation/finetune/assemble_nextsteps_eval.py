"""Assemble a score.py-shaped eval set for the Next Steps task.

`score.py` expects each row to carry `reference_summary`, a `variants` map, and whatever
corpus field the scored variant is grounded against. The Next Steps task has a different
reference (an extracted next-step sentence) and only 75 benchmark calls have one, so this
builds a set it can read without changing the scorer's semantics.

The grounding corpus is the transcript the model actually read -- the `user` field of the
prompt, with the metadata header stripped -- so a next-step sentence that names a person
appears in the corpus only if the transcript does.

    python -I assemble_nextsteps_eval.py --samples nextsteps_samples.jsonl
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "nextsteps_data", "eval.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    by_id = {r["call_id"]: r for r in
             (json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip())}

    want = {r["call_id"] for r in rows}
    missing = sorted(want - set(by_id))
    extra = sorted(set(by_id) - want)
    if missing or extra:
        raise SystemExit(f"join is not exact: {len(missing)} missing {missing[:5]}, "
                         f"{len(extra)} extra {extra[:5]}")
    empty = [c for c in want if not (by_id[c].get("summary") or "").strip()]
    if empty:
        raise SystemExit(f"{len(empty)} empty summaries, e.g. {empty[:5]}")

    out = []
    for r in rows:
        user = r["user"]
        # The prompt is "<metadata header>Transcript:\n\n<body>"; ground on the body.
        body = user.split("Transcript:\n\n", 1)[-1]
        out.append({
            "call_id": r["call_id"],
            "reference_summary": r["reference"],
            "variants": {"student_nextsteps": by_id[r["call_id"]]["summary"].strip()},
            "input_transcript": body,
            "cleaned_transcript": body,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[nextsteps-eval] {len(out)} calls -> {args.out}")
    L = sorted(len(r["variants"]["student_nextsteps"].split()) for r in out)
    print(f"  generated words: median {L[len(L)//2]} min {L[0]} max {L[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
