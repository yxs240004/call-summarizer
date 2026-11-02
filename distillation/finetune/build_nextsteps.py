"""Build the Next Steps task: labels, training rows, and held-out eval references.

The roadmap's "rinse and repeat with different target outputs" starts here. Unlike the
summarizer, this task has no purpose-built target column, so the labels are *extracted*
from text the project already has:

  human    : the "Next step(s):" sentence inside `dataset_raw.json`'s `target_summary`
  teacher  : the same sentence inside `teacher_output.jsonl`, used only where the human
             reference has none (a pseudo-label, the same relationship distillation uses)

Human is preferred wherever both exist -- 41 of 116 training calls have both, and the
eval references are human, so the teacher labels must not set the style the model is
judged against.

Evaluation is honest about its size: only **75** of the 144 benchmark calls have a human
next-step sentence, so this task is scored on 75 calls rather than 144. That is stated in
the score, not hidden by padding.

Input format matches the winning configuration from DESIGN.md step 13 (prospect name, deal
stage, deal outcome, transcript). The task prompt is new: the summarizer's prompt asks for
a briefing, this one asks only for the agreed actions.

    python -I build_nextsteps.py
"""
import argparse
import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "teacher_input.jsonl")
TEACHER = os.path.join(HERE, "teacher_output.jsonl")
RAW = os.path.join(HERE, "dataset_raw.json")
EVAL_IN = os.path.join(HERE, "infer_input_meta.jsonl")
EVAL_SET = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir,
                                        "call-summarizer", "eval", "eval_set.jsonl"))

SYSTEM = (
    "You are a sales-call analyst for an oil & gas acquisitions firm. Read the call "
    "transcript and state the agreed next steps as one factual, third-person sentence for "
    "the rep's CRM. Rules: say who will do what, and by when if the call names a time; use "
    "the prospect's real name and refer to the salesperson as 'the rep' (never "
    "'SPEAKER_00'); do not invent actions, dates or commitments that are not in the "
    "transcript; if no next step was agreed, say so plainly. Aim for 15-60 words."
)

# The next-step sentence ends where the summary's next labelled section begins. The
# terminator is captured so it survives the cut: stripping it taught the model to emit
# sentences that never end, which is what the first pilot run produced.
CUT = re.compile(r"(?P<end>[.;])\s*(?:Outcome signal|Signals?|Stage)\b", re.I)
NEXT = re.compile(r"next steps?\s*:\s*(.+)", re.I | re.S)


def extract_next_steps(text):
    m = NEXT.search(text or "")
    if not m:
        return None
    tail = m.group(1)
    c = CUT.search(tail)
    if c:
        tail = tail[: c.start("end") + 1]
    tail = " ".join(tail.split()).rstrip(" ;")
    if tail and tail[-1] not in ".!?":
        tail += "."
    return tail if len(tail.split()) >= 5 else None


def as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            o = ast.literal_eval(v)
            return o if isinstance(o, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "nextsteps_data"))
    ap.add_argument("--val", type=int, default=12)
    args = ap.parse_args()

    raw = {r["call_id"]: r for r in json.load(open(RAW, encoding="utf-8"))}
    eval_ids = {json.loads(l)["call_id"] for l in open(EVAL_SET, encoding="utf-8") if l.strip()}
    prompts = [json.loads(l) for l in open(PROMPTS, encoding="utf-8") if l.strip()]
    teacher = {json.loads(l)["call_id"]: json.loads(l)["summary"]
               for l in open(TEACHER, encoding="utf-8") if l.strip()}

    # Training pool: the prepared prompts, which are already disjoint from the eval set.
    if any(p["call_id"] in eval_ids for p in prompts):
        raise SystemExit("refusing: a training prompt id is in the eval set")

    rows, sources, problems = [], {}, []
    for p in prompts:
        cid = p["call_id"]
        human = extract_next_steps((raw.get(cid) or {}).get("target_summary"))
        tchr = extract_next_steps(teacher.get(cid))
        target, src = (human, "human") if human else (tchr, "teacher") if tchr else (None, None)
        if not target:
            continue
        d = raw.get(cid) or {}
        names = [a.get("name") for a in as_list(d.get("attendees"))
                 if isinstance(a, dict) and a.get("name")]
        outcome = (d.get("deal_outcome") or "").strip()
        m = re.search(r"Transcript:\n\n(.+)$", p["user"], re.S)
        stage = re.search(r"Deal stage:\s*(.+)", p["user"])
        if not m or not stage or not names or not outcome:
            problems.append(f"{cid}: missing transcript/stage/name/outcome")
            continue
        header = (f"Prospect: {names[0]}\nDeal stage: {stage.group(1).strip()}\n"
                  f"Deal outcome: {outcome}\n\nTranscript:\n\n")
        rows.append({"call_id": cid,
                     "messages": [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": header + m.group(1)},
                                  {"role": "assistant", "content": target}]})
        sources[cid] = src

    if problems:
        print("[nextsteps] refusing:")
        for x in problems[:10]:
            print("   -", x)
        return 1

    # Deterministic split: sorted by call_id, so reruns produce the same partition.
    rows.sort(key=lambda r: r["call_id"])
    if args.val * 2 > len(rows):
        raise SystemExit("not enough rows for a val split")
    val_ids = {r["call_id"] for r in rows[: args.val]}
    train = [r for r in rows if r["call_id"] not in val_ids]
    val = [r for r in rows if r["call_id"] in val_ids]

    # Eval: the 75 benchmark calls whose human reference carries a next-step sentence.
    ev_in = {r["call_id"]: r for r in
             (json.loads(l) for l in open(EVAL_IN, encoding="utf-8") if l.strip())}
    ev_rows, missing_ref = [], 0
    for cid in sorted(ev_in):
        ref = extract_next_steps((raw.get(cid) or {}).get("target_summary"))
        if not ref:
            missing_ref += 1
            continue
        ev_rows.append({"call_id": cid, "system": SYSTEM,
                        "user": ev_in[cid]["user"], "reference": ref})
    if not ev_rows:
        raise SystemExit("refusing: no eval calls have a next-step reference")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (("train", train), ("val", val)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5s}: {len(data):3d} rows -> {path}")
    path = os.path.join(args.out_dir, "eval.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in ev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"eval : {len(ev_rows):3d} rows -> {path}  ({missing_ref} benchmark calls have no human next-step reference)")

    src = {"human": sum(1 for v in sources.values() if v == "human"),
           "teacher": sum(1 for v in sources.values() if v == "teacher")}
    print(f"\nlabels: {src['human']} human, {src['teacher']} teacher-pseudo  (total {len(rows)})")
    print("integrity: train and val ids disjoint:", not ({r['call_id'] for r in train} & val_ids))
    print("integrity: no eval id in training:", not ({r['call_id'] for r in rows} & {r['call_id'] for r in ev_rows}))
    print("\nexample target:", json.dumps(train[0]["messages"][2]["content"][:110]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
