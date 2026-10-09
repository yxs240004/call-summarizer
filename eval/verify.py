"""Deterministic faithfulness gate: flag every figure and proper noun in a
summary that is not recognisably present in the transcript it came from.

This is the run-time companion to score.py's grounding metric. Before a summary
goes into the CRM, run it through here and suppress or review anything flagged.
Reuses score.py's numbers() (spoken-number aware: "nine and a half percent" ==
"9.5%") and name_grounded() (ASR-spelling tolerant: "energy line" == "EnergyLink").
Both still over-report -- a figure in an uncovered form or a name the ASR mangled
past a five-letter prefix reads as ungrounded -- so treat the output as a review
queue, not a verdict.

Reads eval/eval_set.jsonl by default. `--gate` exits 1 if anything is flagged,
so it can gate a write pipeline.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import numbers, _num_display, _name_candidates, name_grounded, _norm_name


def verify_one(summary, transcript):
    """Figures and proper nouns in `summary` that are absent from `transcript`."""
    src_norm = _norm_name(transcript)
    bad_numbers = numbers(summary) - numbers(transcript)
    bad_names = [n for n in _name_candidates(summary) if not name_grounded(n, src_norm)]
    return {
        "ungrounded_numbers": sorted(_num_display(v, k) for k, v in bad_numbers),
        "ungrounded_names": bad_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=os.path.join(HERE, "eval_set.jsonl"))
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if any call has an ungrounded figure or name")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_set, encoding="utf-8") if l.strip()]
    total = 0
    for r in rows:
        corpus = r.get("model_input") or r["cleaned_transcript"]
        for variant, summary in r["variants"].items():
            bad = verify_one(summary, corpus)
            n = len(bad["ungrounded_numbers"]) + len(bad["ungrounded_names"])
            total += n
            if n:
                print(f"{r['call_id']} {variant:<14} figures={bad['ungrounded_numbers']} "
                      f"names={bad['ungrounded_names']}")
    print(f"\n{total} ungrounded items across {len(rows)} calls")
    if args.gate and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
