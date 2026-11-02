"""Summarise one arm's generated samples, so a bad arm is obvious before scoring.

    python sample_stats.py student-samples/blend.jsonl
"""
import json
import sys


def main():
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    tok = sorted(r["summary_tokens"] for r in rows)
    chars = sorted(len(r["summary"]) for r in rows)
    empties = [r["call_id"] for r in rows if not (r.get("summary") or "").strip()]
    mid = [r["call_id"] for r in rows
           if (r.get("summary") or "").rstrip()[-1:] not in ('.', '!', '?', ')')]
    dup = [r["call_id"] for r in rows if r["summary"].count(r["summary"][:60]) > 1]
    name = path.rsplit("/", 1)[-1]
    print(f"[{name}] n={len(rows)} tok p50={tok[len(tok) // 2]} max={tok[-1]} "
          f"chars p50={chars[len(chars) // 2]} min={chars[0]} max={chars[-1]}")
    print(f"[{name}] empty={len(empties)} mid_sentence={len(mid)} repeating_head={len(dup)}")
    for label, ids in (("EMPTY", empties), ("MID-SENTENCE", mid), ("REPEATING", dup)):
        if ids:
            print(f"[{name}] {label}: {ids[:10]}")


if __name__ == "__main__":
    main()
