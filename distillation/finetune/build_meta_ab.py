"""Build the metadata A/B: two student arms that differ only in the user message.

The roadmap question (2026-10-22): does giving the student structured call metadata help
it reproduce the teacher's summaries, or is the transcript enough? The diagram calls for
"extra inputs (speaker name and deal size / deal info)" against "transcript only", and
this builds exactly those two arms against the *same* targets.

  plain : "Transcript:\\n\\n" + body
  meta  : "Prospect: <name>\\nDeal stage: <stage>\\nDeal outcome: <outcome>\\n\\n"
          "Transcript:\\n\\n" + body

Everything else is held fixed and asserted fixed: the system prompt, the target text at
every index, the row count, and the call_id at every index. If any of those diverge the
script refuses to write rather than emit an A/B that varies two things at once.

Where the metadata comes from
-----------------------------
The stage is already inside `teacher_input.jsonl`'s user string ("Deal stage: ..."), so
it is lifted from there rather than re-joined from the raw table, which keeps the `meta`
arm a strict superset of what every existing arm saw. The prospect name and the deal
outcome come from `dataset_raw.json` (`attendees`, `deal_outcome`) and are present for
all 154 calls -- the title is literally "Call with <prospect>" for 153 of them.

Deal *size* is not a field anywhere in the corpus. Rather than invent a proxy (the largest
dollar figure is in the transcript the model already reads), the arm carries the three
metadata fields that genuinely exist. That is a deliberate narrowing, recorded here so a
later reader does not assume the "deal size" half of the diagram was tested.

Layout
------
Rows are laid out exactly as `build_student_data.py` lays out the blend arm: 142 train
calls, each contributing a reference-targeted row and a teacher-targeted row, interleaved
so index i is a reference row and index 2i+1 is that call's teacher row. The validation
split is 12 rows from their reference targets. Because the layout is identical across
arms, index i means the same call and the same target in both files -- which is what makes
the pair comparable at all.

    python -I build_meta_ab.py
    python -I build_meta_ab.py --out-dir student_data
"""
import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(HERE, "teacher_input.jsonl")
DEFAULT_TEACHER = os.path.join(HERE, "teacher_output.jsonl")
DEFAULT_RAW = os.path.join(HERE, "dataset_raw.json")
DEFAULT_OUT_DIR = os.path.join(HERE, "student_data")
ARMS = ("plain", "meta")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_list(v):
    """`attendees` is a real list in the rows this project reads, but older exports
    wrote it as a repr string. Accept both rather than silently dropping the name."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            o = ast.literal_eval(v)
            return o if isinstance(o, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def split_user(user):
    """Return (stage_line, transcript_body).

    The prompts were rendered with `Deal stage: ...` followed by a blank line, the label
    `Transcript:`, a blank line, then the role-mapped turns. Splitting on the labelled
    boundary keeps the transcript byte-identical to what every earlier arm trained on.
    """
    marker = "\n\nTranscript:\n\n"
    if marker not in user:
        raise ValueError("prompt has no 'Transcript:' boundary")
    head, body = user.split(marker, 1)
    if not body.strip():
        raise ValueError("empty transcript body")
    if not head.startswith("Deal stage:"):
        raise ValueError(f"unexpected prompt head: {head[:60]!r}")
    return head, body


def user_for(arm, stage_line, body, name, outcome):
    if arm == "plain":
        return "Transcript:\n\n" + body
    if arm == "meta":
        return (f"Prospect: {name}\n{stage_line}\nDeal outcome: {outcome}\n\n"
                "Transcript:\n\n" + body)
    raise ValueError(arm)


def row(system, user, target, call_id, kind, arm):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "call_id": call_id,
        "kind": kind,
        "arm": arm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    prompts = load_jsonl(args.prompts)
    teacher = {r["call_id"]: r["summary"] for r in load_jsonl(args.teacher)}
    raw = {r["call_id"]: r for r in json.load(open(args.raw, encoding="utf-8"))}

    train = [p for p in prompts if p["split"] == "train"]
    val = [p for p in prompts if p["split"] == "val"]
    if not train or not val:
        raise SystemExit("refusing: one of the splits is empty")
    if {p["call_id"] for p in train} & {p["call_id"] for p in val}:
        raise SystemExit("refusing: train and val ids overlap")

    # Resolve every call's metadata once, and fail loudly on a missing join.
    meta = {}
    problems = []
    for p in prompts:
        cid = p["call_id"]
        if cid not in teacher:
            problems.append(f"{cid}: no teacher summary")
            continue
        r = raw.get(cid) or {}
        names = [a.get("name") for a in as_list(r.get("attendees"))
                 if isinstance(a, dict) and a.get("name")]
        if not names:
            problems.append(f"{cid}: no attendee name")
        outcome = (r.get("deal_outcome") or "").strip()
        if not outcome:
            problems.append(f"{cid}: no deal outcome")
        try:
            stage_line, body = split_user(p["user"])
        except ValueError as e:
            problems.append(f"{cid}: {e}")
            continue
        meta[cid] = {"name": names[0] if names else "", "outcome": outcome,
                     "stage_line": stage_line, "body": body}
    if problems:
        print("[meta-ab] refusing to build:")
        for x in problems[:20]:
            print("   -", x)
        if len(problems) > 20:
            print(f"   ... and {len(problems) - 20} more")
        return 1

    def build(calls, interleave=True):
        """Train rows interleave reference and teacher targets, exactly as
        build_student_data.py does. Validation rows use the reference target only, also
        matching it -- a val loss that included teacher rows would not be comparable to
        the blend arm's."""
        rows = []
        for p in calls:
            cid = p["call_id"]
            pairs = ([("reference", p["reference"]), ("teacher", teacher[cid])]
                     if interleave else [("reference", p["reference"])])
            for kind, target in pairs:
                rows.append((cid, kind, p["system"], target))
        return rows

    train_pairs = build(train)
    val_pairs = build(val, interleave=False)

    out = {}
    for arm in ARMS:
        out[arm] = {
            "train": [row(s, user_for(arm, meta[c]["stage_line"], meta[c]["body"],
                                      meta[c]["name"], meta[c]["outcome"]),
                          t, c, k, arm) for c, k, s, t in train_pairs],
            "val": [row(s, user_for(arm, meta[c]["stage_line"], meta[c]["body"],
                                    meta[c]["name"], meta[c]["outcome"]),
                        t, c, k, arm) for c, k, s, t in val_pairs],
        }

    # --- integrity: the arms must differ in the user message and nowhere else ---------
    for split in ("train", "val"):
        a, b = out["plain"][split], out["meta"][split]
        if len(a) != len(b):
            raise SystemExit(f"[{split}] row counts differ: {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x["call_id"] != y["call_id"]:
                raise SystemExit(f"[{split}:{i}] call_id differs")
            if x["kind"] != y["kind"]:
                raise SystemExit(f"[{split}:{i}] kind differs")
            if x["messages"][0] != y["messages"][0]:
                raise SystemExit(f"[{split}:{i}] system prompt differs")
            if x["messages"][2] != y["messages"][2]:
                raise SystemExit(f"[{split}:{i}] target differs -- the A/B would be confounded")
            if x["messages"][1]["content"] == y["messages"][1]["content"]:
                raise SystemExit(f"[{split}:{i}] user messages are identical -- the arm is a no-op")
            if not x["messages"][2]["content"].strip():
                raise SystemExit(f"[{split}:{i}] empty target")

    os.makedirs(args.out_dir, exist_ok=True)
    for arm in ARMS:
        for split in ("train", "val"):
            path = os.path.join(args.out_dir, f"{arm}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in out[arm][split]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{arm:<6} {split:<5}: {len(out[arm][split]):>4} rows -> {path}")

    # A short, human-readable diff of one call so the change is visible, not assumed.
    cid = train_pairs[0][0]
    print(f"\nexample ({cid}):")
    print("  plain user:", json.dumps(out["plain"]["train"][0]["messages"][1]["content"][:90]))
    print("  meta  user:", json.dumps(out["meta"]["train"][0]["messages"][1]["content"][:90]))
    print(f"  target    : {json.dumps(out['plain']['train'][0]['messages'][2]['content'][:70])}")
    print("\nintegrity: identical system prompt, target and call order; user differs on every row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
