"""Build the teacher's inference prompts for the holdout training split.

Step 1 of the distillation build (see DESIGN.md). The teacher is the 8B LoRA in
this directory; the split is the 154-record holdout -- 142 train + 12 val -- whose
record ids are disjoint from the 144-call benchmark, so every pseudo-label is
generated on a call the teacher was trained on but the benchmark does not grade.

The prompts are read straight out of the holdout JSONL rather than re-derived from
dataset_raw.json. This is deliberate. Those rows already hold build_model_input()
output verbatim, so reading them guarantees the teacher sees byte-for-byte what the
8B was trained on; re-deriving would risk a silent divergence the moment
build_model_input() changes.

Those rows carry NO identifier -- keys are exactly {"messages"} -- so identity is
recovered by joining each row's assistant target back against dataset_raw.json's
target_summary. All 305 target summaries are distinct, so the join is unambiguous.
The script refuses to write anything if even one row fails to join: a mislabelled
pseudo-label is worse than no pseudo-label, and a silent join failure would corrupt
the training set without any visible symptom.

Writes teacher_input.jsonl, one line per record:
    {call_id, split, title, system, user, reference}

The reference summary rides along so the student's blend dataset can be built from
this single file, without a second join and without touching the holdout dir again.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_DIR = r"C:\Projects\call-summarizer\finetune"
DEFAULT_RAW = os.path.join(HERE, "dataset_raw.json")
DEFAULT_OUT = os.path.join(HERE, "teacher_input.jsonl")
SPLITS = ("train", "val")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_target(raw):
    """Map target_summary text -> [record, ...]. More than one record per target
    means the join is ambiguous and the script must not proceed."""
    index = {}
    for rec in raw:
        target = rec.get("target_summary")
        if target:
            index.setdefault(target, []).append(rec)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR,
                    help="directory holding the holdout train.jsonl and val.jsonl")
    ap.add_argument("--raw", default=DEFAULT_RAW,
                    help="dataset_raw.json supplying call_id and title")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the rows that DID join instead of failing")
    args = ap.parse_args()

    raw = json.load(open(args.raw, encoding="utf-8"))
    index = index_by_target(raw)
    ambiguous = [t for t, recs in index.items() if len(recs) > 1]
    print(f"index      : {len(raw)} raw records -> {len(index)} distinct targets")
    if ambiguous:
        print(f"AMBIGUOUS  : {len(ambiguous)} target summaries map to >1 record — "
              f"the join cannot be trusted")

    rows, problems = [], []
    for split in SPLITS:
        path = os.path.join(args.source_dir, f"{split}.jsonl")
        for i, row in enumerate(load_jsonl(path)):
            msgs = row["messages"]
            if [m["role"] for m in msgs] != ["system", "user", "assistant"]:
                problems.append((split, i, f"unexpected roles {[m['role'] for m in msgs]}"))
                continue
            system, user, reference = (m["content"] for m in msgs)

            hits = index.get(reference, [])
            if len(hits) != 1:
                problems.append((split, i, f"{len(hits)} records match this target"))
                continue

            rows.append({
                "call_id": hits[0]["call_id"],
                "split": split,
                "title": hits[0].get("title"),
                "system": system,
                "user": user,
                "reference": reference,
            })

    # --- integrity gates, all of which must pass before anything is written ---
    systems = {r["system"] for r in rows}
    ids = [r["call_id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}

    print(f"joined     : {len(rows)} rows "
          f"({sum(r['split'] == 'train' for r in rows)} train / "
          f"{sum(r['split'] == 'val' for r in rows)} val)")
    print(f"system     : {len(systems)} distinct system prompt(s)")
    print(f"call ids   : {len(set(ids))} distinct")
    if dupes:
        print(f"DUPLICATES : {len(dupes)} call_id(s) appear more than once")
    if problems:
        print(f"UNJOINED   : {len(problems)} row(s)")
        for split, i, why in problems:
            print(f"  {split}[{i}]: {why}")

    if ambiguous or dupes or problems:
        if not args.allow_missing:
            raise SystemExit(
                "\nrefusing to write: the join is not clean. Fix the cause, or pass "
                "--allow-missing if you have read the report above and accept it."
            )

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(len(r["user"]) for r in rows)
    print(f"\nwrote {len(rows)} prompts -> {args.out}")
    print(f"prompt size: {total_chars:,} chars total, "
          f"{total_chars // max(len(rows), 1):,} chars average")


if __name__ == "__main__":
    main()
