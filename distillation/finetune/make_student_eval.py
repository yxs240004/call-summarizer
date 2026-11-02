"""Add the student's summaries to the frozen eval set as extra variants.

Step 5 of the distillation build (see DESIGN.md D8). Rather than reimplement ROUGE,
this copies `call-summarizer/eval/eval_set.jsonl` and adds one variant per arm, so
`eval/score.py` reports the student in the same table, from the same references,
under the same bootstrap as `qwen4b` 0.4410 and `qwen8b_honest` 0.4372. A number
produced any other way would not be comparable to those two, and comparability is
the only reason to run the eval at all.

The join is on call_id and is asserted to be exact in both directions: every eval
call must get a student summary, and the student must not contain a call the eval
set does not know about. A partial join would silently score fewer calls and shift
the mean without any visible error.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL = os.path.join(os.path.dirname(HERE), "..", "call-summarizer", "eval", "eval_set.jsonl")
DEFAULT_OUT = os.path.join(HERE, "student_eval.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-set", default=os.path.abspath(DEFAULT_EVAL))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--samples", nargs="+", required=True,
                    help="arm=path pairs, e.g. blend=student-samples/blend.jsonl")
    ap.add_argument("--only", nargs="+", default=None,
                    help="keep only these variants in the output. score.py emits its "
                         "paired bootstrap only with exactly two variants, so pass two "
                         "(e.g. --only student_blend qwen4b) to get a significance test "
                         "rather than two overlapping confidence intervals.")
    ap.add_argument("--inputs", nargs="+", default=[],
                    help="arm=path pairs naming the prompt file that arm actually read. "
                         "Stored on every row as input_<arm>, so score.py can ground that "
                         "arm against its own input instead of a corpus it never saw. "
                         "Needed when arms differ in their prompt, not just their weights.")
    args = ap.parse_args()

    inputs = {}
    for spec in args.inputs:
        if "=" not in spec:
            sys.exit(f"--inputs entry must be arm=path, got {spec!r}")
        arm, path = spec.split("=", 1)
        inputs[arm] = {r["call_id"]: r["user"] for r in load_jsonl(path)}
        print(f"[eval] input for {arm}: {len(inputs[arm])} prompts from {path}")

    if not os.path.exists(args.eval_set):
        sys.exit(f"eval set not found: {args.eval_set}")
    rows = load_jsonl(args.eval_set)
    want = {r["call_id"] for r in rows}
    print(f"[eval] {len(rows)} calls, existing variants: "
          f"{sorted({v for r in rows for v in r.get('variants', {})})}")

    for spec in args.samples:
        if "=" not in spec:
            sys.exit(f"--samples entry must be arm=path, got {spec!r}")
        arm, path = spec.split("=", 1)
        samples = load_jsonl(path)
        by_id = {}
        for s in samples:
            if s["call_id"] in by_id:
                sys.exit(f"{arm}: duplicate call_id {s['call_id']} in {path}")
            by_id[s["call_id"]] = s

        missing = sorted(want - set(by_id))
        extra = sorted(set(by_id) - want)
        if missing or extra:
            sys.exit(f"{arm}: join is not exact -- {len(missing)} eval call(s) have no "
                     f"summary {missing[:5]}, {len(extra)} sample(s) are not eval calls "
                     f"{extra[:5]}. Scoring a subset would move the mean silently.")
        empty = [c for c in want if not (by_id[c].get("summary") or "").strip()]
        if empty:
            sys.exit(f"{arm}: {len(empty)} empty summar(ies), e.g. {empty[:5]}")

        variant = f"student_{arm}"
        prompt = inputs.get(arm)
        if prompt is not None:
            miss = sorted(want - set(prompt))
            if miss:
                sys.exit(f"{arm}: {len(miss)} eval call(s) have no prompt {miss[:5]}")
        for r in rows:
            r.setdefault("variants", {})[variant] = by_id[r["call_id"]]["summary"].strip()
            if prompt is not None:
                r[f"input_{arm}"] = prompt[r["call_id"]]
        lengths = sorted(len(by_id[c]["summary"]) for c in want)
        print(f"[eval] added {variant}: {len(want)} calls, char p50 {lengths[len(lengths) // 2]}, "
              f"min {lengths[0]}, max {lengths[-1]}")

    if args.only:
        missing = [v for v in args.only if v not in {v for r in rows for v in r["variants"]}]
        if missing:
            sys.exit(f"--only names variant(s) that are not present: {missing}")
        kept = 0
        for r in rows:
            r["variants"] = {k: v for k, v in r["variants"].items() if k in args.only}
            kept += len(r["variants"])
        print(f"[eval] restricted to {args.only} ({kept} variant rows) "
              f"-- score.py will emit a paired bootstrap")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {args.out}\n"
          f"[eval] score with: python score.py --eval-set {args.out} --frozen EVAL_FROZEN.json")


if __name__ == "__main__":
    main()
