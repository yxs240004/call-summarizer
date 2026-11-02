"""Turn dataset_raw.json into chat-format training/validation JSONL.

Each example becomes a 3-message chat: system (house-style instructions),
user (the transcript + metadata, built by build_model_input), assistant
(the existing Supabase summary = the target style).

Run AFTER the TODO(human) below is filled in.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "dataset_raw.json")
TRAIN_PATH = os.path.join(HERE, "train.jsonl")
VAL_PATH = os.path.join(HERE, "val.jsonl")
VAL_FRACTION = 0.08
SEED = 13

# Longest transcripts are ~29k chars (~7k tokens). We cap the transcript so every
# training example fits a 4096-token context. 12k chars is a safe round number.
MAX_TRANSCRIPT_CHARS = 12000

# Outlier targets: a handful of the 305 summaries ramble (long + full of
# negation-filler like "no competing vendors", "not discussed"). Training on
# them teaches the model to ramble, so we drop them. Toggle to False to keep all.
FILTER_OUTLIERS = True
MAX_TARGET_CHARS = 1800
MAX_RAMBLE = 4
RAMBLE_MARKERS = [
    "not discussed", "not mentioned", "not confirmed", "no additional",
    "no competing", "no technical", "no contractual", "no legal",
    "no explicit", "not identified", "not raised", "no blockers",
]


def _is_clean(record):
    """False if this target is a rambling outlier we don't want the model to imitate."""
    if not FILTER_OUTLIERS:
        return True
    target = record["target_summary"]
    if len(target) > MAX_TARGET_CHARS:
        return False
    lowered = target.lower()
    return sum(lowered.count(m) for m in RAMBLE_MARKERS) < MAX_RAMBLE

# The house style, encoded as instructions. The LoRA learns the style from the
# assistant targets; this prompt mainly pins the *format* so training and
# inference agree.
SYSTEM_PROMPT = (
    "You are a sales-call analyst for an oil & gas acquisitions firm. "
    "Summarize the call transcript as a factual, third-person briefing for the "
    "rep's CRM. Rules: refer to the prospect by their real name and the "
    "salesperson as 'the rep' (never 'SPEAKER_00'); lead with a narrative of "
    "what was discussed and decided; include concrete specifics (names, dates, "
    "dollar amounts, well counts, terms) when present; then a 'Next steps:' "
    "sentence; then an 'Outcome signal:' sentence framing the deal stage and "
    "engagement. Aim for 500-1200 characters. Do not invent details that are "
    "not in the transcript."
)


def _prospect_name(record):
    """Pull the prospect's real name out of the attendees list.

    attendee entries look like {"name": "Alex Morgan", "email": ...}.
    Falls back to a generic phrase if the list is missing or empty.
    """
    attendees = record.get("attendees") or []
    if attendees and isinstance(attendees[0], dict) and attendees[0].get("name"):
        return attendees[0]["name"]
    return "the prospect"


def build_model_input(record):
    """Render ONE call record into the user-message text the model sees.

    The model must learn to turn "SPEAKER_01" into a real name, so we label each
    turn ourselves. Each segment already tells us its role ("rep"/"prospect").
    """
    prospect_name = _prospect_name(record)

    # --- metadata header ---
    # We include the deal STAGE (their summaries reference it, so it was known
    # to whoever wrote them), but we deliberately do NOT include deal_outcome:
    # that's the "answer" the model is supposed to infer from the call. Flip
    # this if you want to test the other choice.
    header = []
    if record.get("stage_at_call_label"):
        header.append(f"Deal stage: {record['stage_at_call_label']}")

    # --- transcript: one labelled line per spoken turn ---
    lines = []
    for seg in record["segments"]:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # ===================== TODO(human) =====================
        # Decide the LABEL that goes in front of this turn's words.
        # `seg["role"]` is either "prospect" or "rep".
        #   - When the prospect speaks, we want THEIR NAME before the words.
        #   - When the rep speaks, we want the word "rep" before the words.
        # Both of the values you need already exist above: `prospect_name`
        # and the string "rep".
        if seg.get("role") == "prospect":
            label = prospect_name  # prospect speaks -> use their real name
        else:
            label = "rep"          # salesperson speaks -> generic "rep"
        # ========================================================

        lines.append(f"{label}: {text}")
    transcript = "\n".join(lines)

    # --- keep every example inside the context window ---
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n[...transcript truncated...]"

    # --- assemble the final user message ---
    parts = []
    if header:
        parts.append("\n".join(header))
    parts.append("Transcript:")
    parts.append(transcript)
    parts.append(f"(The prospect on this call is {prospect_name}.)")
    return "\n\n".join(parts)


def to_example(record):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_model_input(record)},
        {"role": "assistant", "content": record["target_summary"]},
    ]}


def main():
    records = json.load(open(RAW_PATH, encoding="utf-8"))
    records = [r for r in records if r.get("target_summary") and r.get("segments")]
    before = len(records)
    records = [r for r in records if _is_clean(r)]
    print(f"usable records: {before} | after outlier filter: {len(records)} "
          f"(dropped {before - len(records)})")

    rng = random.Random(SEED)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    val, train = records[:n_val], records[n_val:]

    for path, rows in ((TRAIN_PATH, train), (VAL_PATH, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_example(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")

    ex = to_example(train[0])
    print("\n--- sample USER message (first 700 chars) ---")
    print(ex["messages"][1]["content"][:700])


if __name__ == "__main__":
    main()
