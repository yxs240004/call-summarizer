"""Audit a teacher-output file for degenerate pseudo-labels, before you train on it.

Sequence-level distillation copies the teacher's failures as faithfully as its
successes -- a degenerate label is not a slightly-worse target, it is a target
that teaches the student to repeat itself. So the pass gets audited before it
becomes a dataset.

Two symptoms, because they catch different failures:

  * at the cap   -- generation ran until it was cut off, i.e. the model never
                    chose to stop. The file records no done_reason, so a token
                    count sitting exactly on the cap is the evidence.
  * a repeated   -- the same n-gram many times over. This is the visible shape
    n-gram          of a loop, but it is the weaker detector: a summary emitted
                    twice has every n-gram exactly twice, which a threshold of
                    4 misses. The cap catches that one.

    python -I audit_teacher.py teacher_output.before_redo.jsonl
    python -I audit_teacher.py teacher_output.jsonl            # after the repair
"""
import collections
import json
import sys

CAP = 900
REPEATS = 4
NGRAM = 6


def worst_ngram(text, n=NGRAM):
    """The most-repeated n-gram in a summary, and how many times it occurs."""
    words = text.split()
    grams = collections.Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return "", 0
    gram, times = grams.most_common(1)[0]
    return " ".join(gram), times


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "teacher_output.before_redo.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    lengths = sorted(r["summary_tokens"] for r in rows)
    med = lengths[len(lengths) // 2]
    print(f"{path}: {len(rows)} rows, median {med} tokens, "
          f"max {lengths[-1]} tokens\n")

    flagged = 0
    for row in rows:
        tokens = row["summary_tokens"]
        gram, times = worst_ngram(row["summary"])
        at_cap = tokens >= CAP
        looping = times >= REPEATS
        if at_cap or looping:
            flagged += 1
            mark = "AT CAP" if at_cap else "      "
            print(f"{row['call_id']}  split={row['split']:<5} "
                  f"tokens={tokens:<4} {mark}  {times}x  {gram[:44]!r}")

    print(f"\nflagged {flagged} of {len(rows)} rows")
    if flagged:
        print("a degenerate label is a target that teaches repetition: "
              "regenerate these before they become training data")


if __name__ == "__main__":
    main()
