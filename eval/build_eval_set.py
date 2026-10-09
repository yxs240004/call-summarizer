"""
Build the frozen evaluation set from the artifacts already on disk.

Reads the existing runs rather than re-generating anything:

  reference        _supabase_summaries.json  (the `summmary` column, house style)
  qwen8b_lora      output_v2/<id>.json       (Qwen3-8B + LoRA, trained 2026-10-07)
  bart             output/<id>.json          (knkarthick/MEETING_SUMMARY baseline)
  cleaned          data/cleaned/<id>.txt     (disfluency-stripped transcript)
  model_input      finetune/infer_input.jsonl (the exact user turn the Qwen model saw)

The clean transcripts and model inputs differ on purpose: copy_rate is only
meaningful against the text a given model actually read, so both are carried.

Writes eval/eval_set.jsonl (contains customer data -- gitignored) and
eval/EVAL_FROZEN.json (call ids + sha256 only -- safe to commit).
"""

import argparse
import hashlib
import json
import os
import sys

DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "meeting-summarizer")
)
HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ("qwen8b_lora", "bart")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="directory holding data/, output/, output_v2/, finetune/ "
                         "(default: ../meeting-summarizer)")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_set.jsonl"))
    ap.add_argument("--frozen-out", default=os.path.join(HERE, "EVAL_FROZEN.json"))
    args = ap.parse_args()

    root = args.data_root
    index_path = os.path.join(root, "data", "cleaned_index.csv")
    for rel in ("data/cleaned_index.csv", "output", "output_v2",
                "_supabase_summaries.json", "finetune/infer_input.jsonl"):
        if not os.path.exists(os.path.join(root, rel)):
            sys.exit(f"missing {os.path.join(root, rel)} -- is --data-root right?")

    # call ids, in the order the index records them
    import csv
    with open(index_path, encoding="utf-8-sig") as f:
        ids = [r["call_id"] for r in csv.DictReader(f)]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate call_id in cleaned_index.csv")
    ids = sorted(ids)

    # house reference summaries
    refs = {str(r["call_id"]): r.get("summmary") for r in load_json(
        os.path.join(root, "_supabase_summaries.json"))}

    # exact model input per call (the user turn of the trained chat example)
    model_input = {}
    with open(os.path.join(root, "finetune", "infer_input.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                model_input[row["call_id"]] = row["user"]

    rows, missing, ref_mismatch = [], [], []
    for cid in ids:
        cleaned_p = os.path.join(root, "data", "cleaned", f"{cid}.txt")
        bart_p = os.path.join(root, "output", f"{cid}.json")
        qwen_p = os.path.join(root, "output_v2", f"{cid}.json")
        if not all(os.path.exists(p) for p in (cleaned_p, bart_p, qwen_p)):
            missing.append(cid)
            continue

        with open(cleaned_p, encoding="utf-8-sig") as f:
            cleaned = f.read().strip()

        bart = load_json(bart_p)
        qwen = load_json(qwen_p)
        reference = qwen.get("original_summary") or refs.get(cid)

        # the reference reaching us by two routes must agree
        if refs.get(cid) and qwen.get("original_summary") and refs[cid] != qwen["original_summary"]:
            ref_mismatch.append(cid)

        rows.append({
            "call_id": cid,
            "reference_summary": reference,
            "cleaned_transcript": cleaned,
            "model_input": model_input.get(cid, cleaned),
            "variants": {
                "qwen8b_lora": qwen.get("summary") or "",
                "bart": bart.get("summary") or "",
            },
        })

    if missing:
        sys.exit(f"{len(missing)} call_ids lack an artifact: {missing[:5]}")
    if ref_mismatch:
        sys.exit(f"reference disagrees between the two sources for {ref_mismatch[:5]}")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # freeze on content, so a later change to any artifact is detectable
    canon = json.dumps(
        [{"call_id": r["call_id"], "reference_summary": r["reference_summary"],
          "variants": r["variants"]} for r in rows],
        sort_keys=True, ensure_ascii=False,
    )
    frozen = {
        "n_calls": len(rows),
        "variants": list(VARIANTS),
        "call_ids": [r["call_id"] for r in rows],
        "content_sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest(),
        "reference_source": "_supabase_summaries.json `summmary` (via output_v2 original_summary)",
        "note": "All 144 calls were in the Qwen3-8B-LoRA training rows (268 of 305), "
                "so qwen8b_lora scores here measure style fit, NOT generalization.",
    }
    with open(args.frozen_out, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"frozen sha256 {frozen['content_sha256'][:16]}... -> {args.frozen_out}")


if __name__ == "__main__":
    main()
