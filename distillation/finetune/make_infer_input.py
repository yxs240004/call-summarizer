"""Build the inference prompts for the 144 target calls.

Reuses build_model_input() from build_dataset.py, so the text the model sees at
inference time is byte-for-byte the same shape as what it was trained on. Writes
infer_input.jsonl, which carries the system prompt per line so the pod-side
predict.py needs no access to our data-processing code.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_dataset import SYSTEM_PROMPT, _prospect_name as _name_of, build_model_input  # noqa: E402

RAW_PATH = os.path.join(HERE, "dataset_raw.json")
INDEX_PATH = r"C:\Projects\meeting-summarizer\data\cleaned_index.csv"
CLEANED_DIR = r"C:\Projects\meeting-summarizer\data\cleaned"
OUT_PATH = os.path.join(HERE, "infer_input.jsonl")


def target_ids():
    with open(INDEX_PATH, encoding="utf-8-sig") as f:
        return [row["call_id"] for row in csv.DictReader(f)]


def segments_from_txt(call_id, speaker_mapping):
    """Fallback: rebuild labelled segments from the cleaned .txt + speaker_mapping.

    speaker_mapping maps a speaker index ("0", "3") to a role ("rep"/"prospect").
    """
    path = os.path.join(CLEANED_DIR, f"{call_id}.txt")
    if not os.path.exists(path):
        return []
    mapping = speaker_mapping or {}
    segs = []
    for block in open(path, encoding="utf-8").read().split("\n\n"):
        block = block.strip()
        if not block or ":" not in block:
            continue
        speaker, _, text = block.partition(":")
        idx = speaker.strip().replace("SPEAKER_", "").lstrip("0") or "0"
        role = mapping.get(idx) or mapping.get(speaker.strip())
        segs.append({"speaker": speaker.strip(), "role": role, "text": text.strip().strip('"')})
    return segs


def unlabeled_prompt(call_id, record):
    """Prompt for a call whose transcript has no speaker labels at all.

    One of the 144 has a plain unsegmented blob (no SPEAKER_xx prefixes), so we
    keep the same header / trailer skeleton but drop the per-turn labels rather
    than mislabel every line as "rep".
    """
    text = open(os.path.join(CLEANED_DIR, f"{call_id}.txt"), encoding="utf-8").read().strip()
    if not text:
        return None
    parts = []
    if record.get("stage_at_call_label"):
        parts.append(f"Deal stage: {record['stage_at_call_label']}")
    parts.append("Transcript (unlabeled):")
    parts.append(text[:12000])
    parts.append(f"(The prospect on this call is {record.get('_prospect_name', 'the prospect')}.)")
    return "\n\n".join(parts)


def main():
    records = {r["call_id"]: r for r in json.load(open(RAW_PATH, encoding="utf-8"))}
    ids = target_ids()
    print(f"target calls: {len(ids)}")

    rows, missing = [], []
    for call_id in ids:
        rec = records.get(call_id)
        if rec is None:
            missing.append((call_id, "not in dataset_raw"))
            continue
        if not rec.get("segments"):
            rec = dict(rec, segments=segments_from_txt(call_id, rec.get("speaker_mapping")))
        if rec.get("segments"):
            user = build_model_input(rec)
        else:
            rec = dict(rec, _prospect_name=_name_of(rec))
            user = unlabeled_prompt(call_id, rec)
            if user is None:
                missing.append((call_id, "no segments and no cleaned txt"))
                continue
        rows.append({
            "call_id": call_id,
            "title": rec.get("title"),
            "system": SYSTEM_PROMPT,
            "user": user,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} prompts -> {OUT_PATH}")
    if missing:
        print(f"SKIPPED {len(missing)}:")
        for cid, why in missing:
            print(f"  {cid}: {why}")


if __name__ == "__main__":
    main()
