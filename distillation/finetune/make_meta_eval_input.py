"""Build the two eval prompts for the metadata A/B, byte-matching the training format.

`build_meta_ab.py` emits the training rows for the `plain` and `meta` arms. This emits the
matching *inference* prompts for the same 144 benchmark calls, and it has to agree with the
training render exactly: a prompt shape that differs between training and inference is the
defect this project has already hit twice (a Qwen3 `think` marker, and a Qwen3.5 added-token
pair). So the header is built by the same two lines of formatting, not re-derived here.

  plain : "Transcript:\\n\\n" + body
  meta  : "Prospect: <name>\\nDeal stage: <stage>\\nDeal outcome: <outcome>\\n\\n"
          "Transcript:\\n\\n" + body

Every field is real: the stage is lifted from the prompt the earlier arms already used, and
the prospect name and deal outcome come from `dataset_raw.json`, which contains all 144
benchmark calls (they are held out of the *training* split, not out of the raw table).

    python -I make_meta_eval_input.py
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "infer_input.jsonl")
RAW = os.path.join(HERE, "dataset_raw.json")
# Two marker spellings appear in the 144-call benchmark set: the usual labelled
# transcript, and one call whose turns were never role-mapped. Both are kept verbatim
# so the body is byte-identical to what the earlier arms read.
MARKERS = ("Transcript:\n\n", "Transcript (unlabeled):\n\n")


def split_prompt(user):
    """Return (deal-stage line, transcript body, marker as it appeared).

    The marker is returned verbatim so the `plain` arm reproduces the prompt an earlier
    arm actually read, including the one call whose turns were never role-mapped."""
    for m in MARKERS:
        i = user.find(m)
        if i >= 0:
            return user[:i].rstrip("\n"), user[i + len(m):], m
    raise ValueError("no transcript marker")


def name_from_title(title):
    """Last resort for the one benchmark call with an empty `attendees` list."""
    t = (title or "").strip()
    for suf in (" Discussion", " Zoom Meeting"):
        if t.endswith(suf):
            return t[: -len(suf)].strip()
    if t.startswith("Call with "):
        return t[len("Call with "):].strip()
    return ""


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
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    raw = {r["call_id"]: r for r in json.load(open(RAW, encoding="utf-8"))}

    plain, meta, problems, fallbacks = [], [], [], []
    for i, r in enumerate(rows):
        cid = r["call_id"]
        try:
            head, body, marker = split_prompt(r["user"])
        except ValueError as e:
            problems.append(f"{cid}: {e}")
            continue
        if not head.startswith("Deal stage:"):
            problems.append(f"{cid}: unexpected head {head[:40]!r}")
            continue
        d = raw.get(cid) or {}
        names = [a.get("name") for a in as_list(d.get("attendees"))
                 if isinstance(a, dict) and a.get("name")]
        if names:
            name = names[0]
        else:
            name = name_from_title(r.get("title"))
            fallbacks.append((cid, r.get("title"), name))
        if not name:
            problems.append(f"{cid}: no attendee name and no usable title")
        outcome = (d.get("deal_outcome") or "").strip()
        if not outcome:
            problems.append(f"{cid}: no deal outcome")
        plain.append({"call_id": cid, "system": r["system"], "title": r.get("title"),
                      "user": marker + body})
        meta.append({"call_id": cid, "system": r["system"], "title": r.get("title"),
                     "user": (f"Prospect: {name}\n{head}\n"
                              f"Deal outcome: {outcome}\n\n" + marker + body)})

    if problems:
        print("[meta-eval] refusing to build:")
        for p in problems[:20]:
            print("   -", p)
        return 1

    if fallbacks:
        print(f"[meta-eval] name taken from the title on {len(fallbacks)} call(s):")
        for cid, title, name in fallbacks:
            print(f"   - {cid}: title={title!r} -> Prospect: {name!r}")

    for name, data in (("infer_input_plain.jsonl", plain),
                       ("infer_input_meta.jsonl", meta)):
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} prompts")

    # The two files must differ only inside the metadata header of the user message.
    for i, (a, b) in enumerate(zip(plain, meta)):
        if a["call_id"] != b["call_id"] or a["system"] != b["system"]:
            raise SystemExit(f"row {i}: id or system prompt diverged")
        if split_prompt(a["user"])[1] != split_prompt(b["user"])[1]:
            raise SystemExit(f"row {i}: transcript body diverged")
        if a["user"] == b["user"]:
            raise SystemExit(f"row {i}: prompts identical -- the arm is a no-op")

    print("\nexample user (meta):")
    print("  " + json.dumps(meta[0]["user"][:120]))
    print("integrity: same ids, same order, same transcript body; only the header differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
