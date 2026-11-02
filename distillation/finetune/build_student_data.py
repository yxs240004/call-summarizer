"""Build the student's training data: two arms, matched at 284 examples each.

Step 3 of the distillation build (see DESIGN.md D5/D6).

    blend    : 142 reference-targeted + 142 teacher-targeted  = 284
    control  : 142 reference-targeted, duplicated once        = 284

The point of the control is that it differs from the blend in exactly one way --
what the second half of its data contains -- and in nothing else. Same row count,
same step count, same epochs. If the control also saw half as much data, a
difference between the arms could just be "more data won", and the comparison
would say nothing about distillation.

Only the 142 *train* calls are built here. The 12 validation calls are emitted
once, from their reference targets, and are byte-identical in both arms, so they
can never contribute to a difference between the arms.

Output rows are {"messages": [system, user, assistant]} -- the shape train.py reads and
the shape train_student.py reads. call_id and target ride along so a later failure can
be traced to the row that caused it; train_student.py uses only messages and call_id.

Integrity is checked before anything is written, and the script refuses to write
if a check fails. A silent join failure here would train the student on a
mislabelled target with no visible symptom until evaluation.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(HERE, "teacher_input.jsonl")
DEFAULT_TEACHER = os.path.join(HERE, "teacher_output.jsonl")
DEFAULT_OUT_DIR = os.path.join(HERE, "student_data")
ARMS = ("blend", "control")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row(system, user, target, call_id, kind):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "call_id": call_id,
        "target": kind,
    }


def build_train(prompts, teacher, arm, allow_empty):
    """The 142 training rows for one arm, reference and teacher rows interleaved.

    Interleaved rather than concatenated: with per_device_train_batch_size=1 and a
    shuffled sampler it makes little difference to the optimiser, but it keeps the
    two arms' rows in the same order, so a diff of the two files shows the arms
    differing only in the second half's *content* -- which is the whole claim.
    """
    by_id = {t["call_id"]: t for t in teacher}
    missing = [p["call_id"] for p in prompts if p["call_id"] not in by_id]
    if missing:
        raise SystemExit(f"refusing to write: {len(missing)} call(s) have no teacher "
                         f"output: {missing[:10]}")

    rows = []
    for p in prompts:
        rows.append(row(p["system"], p["user"], p["reference"], p["call_id"], "reference"))
        t = by_id[p["call_id"]]
        summary = (t.get("summary") or "").strip()
        if not summary and not allow_empty:
            raise SystemExit(
                f"refusing to write: teacher summary for {p['call_id']} is empty. An "
                f"empty target teaches the student to emit nothing. Re-run that call, "
                f"or pass --allow-empty if you have read the teacher log and accept it."
            )
        rows.append(row(p["system"], p["user"], summary, p["call_id"], "teacher"))

    if arm == "control":
        # The reference half, duplicated in place: [ref_0, ref_0, ref_1, ref_1, ...].
        #
        # Duplicated *in place*, not concatenated. Concatenating ([refs, refs]) also
        # reaches 284 rows, but it moves every reference row to a different index than
        # the blend gives it, so the two arms would differ in more than the content of
        # the odd rows. Keeping the layout identical means index i is a reference in
        # both arms and only the odd indices differ -- teacher summary in the blend, a
        # repeat of that call's reference in the control. That is the one difference the
        # comparison needs, and nothing else.
        refs = rows[: 2 * len(prompts): 2]
        rows = [r for r in refs for _ in range(2)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--allow-empty", action="store_true",
                    help="keep empty teacher summaries instead of failing")
    args = ap.parse_args()

    all_prompts = load_jsonl(args.prompts)
    teacher = load_jsonl(args.teacher)

    train_prompts = [p for p in all_prompts if p["split"] == "train"]
    val_prompts = [p for p in all_prompts if p["split"] == "val"]
    print(f"prompts    : {len(all_prompts)} ({len(train_prompts)} train / "
          f"{len(val_prompts)} val)")
    print(f"teacher    : {len(teacher)} summaries")

    # Gates that must hold before the arms mean anything.
    train_ids = {p["call_id"] for p in train_prompts}
    val_ids = {p["call_id"] for p in val_prompts}
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f"refusing to write: {len(overlap)} call(s) in both splits: "
                         f"{sorted(overlap)[:10]}")
    if len(train_ids) != len(train_prompts):
        raise SystemExit("refusing to write: duplicate call_id within the train split")
    dup_teacher = len(teacher) - len({t["call_id"] for t in teacher})
    if dup_teacher:
        print(f"WARNING    : {dup_teacher} duplicate call_id(s) in the teacher output; "
              f"the last one written wins")

    val_rows = [row(p["system"], p["user"], p["reference"], p["call_id"], "reference")
                for p in val_prompts]

    os.makedirs(args.out_dir, exist_ok=True)
    for arm in ARMS:
        rows = build_train(train_prompts, teacher, arm, args.allow_empty)
        kinds = {}
        for r in rows:
            kinds[r["target"]] = kinds.get(r["target"], 0) + 1
        path = os.path.join(args.out_dir, f"{arm}_train.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{arm:<9}: {len(rows)} rows {kinds} -> {path}")

    val_path = os.path.join(args.out_dir, "val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"val      : {len(val_rows)} rows -> {val_path} (shared by both arms)")

    # The arms must be the same length, or a difference between them is not a
    # distillation effect. Cheap to assert, expensive to get wrong.
    lengths = {arm: len(build_train(train_prompts, teacher, arm, args.allow_empty))
               for arm in ARMS}
    print(f"\narm sizes: {lengths} -- matched: {len(set(lengths.values())) == 1}")


if __name__ == "__main__":
    main()
