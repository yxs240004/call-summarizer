"""Summarize ONE call transcript with the distilled student, and show your work.

    python -I try_call.py mycall.txt --prospect "Sam Whitfield" --stage "Discovery/Qualification"

This is the demo path, not the measurement path. It runs the artifact that actually ships
(student-blend, the 1.27 GB Q4_K_M file) through the same Ollama options the scored runs
used, and prints the summary, the timing, and the raw prompt so you can see what the model
was given.

WHY IT DOES NOT HAND-BUILD THE PROMPT. The render comes from student_generate.render(),
which calls the adapter's own tokenizer, and the system prompt is read out of
infer_input.jsonl rather than retyped here. Both choices are deliberate: this project has
already lost a run to a chat template that was transcribed by hand, and the two renders
differed on one token that is invisible in a printed string. A prompt this file composed
by string concatenation would be a different prompt from the one the student trained on,
and nothing would say so.

The decoding options come from measure_local.options() for the same reason -- one place
defines what "the shipped configuration" means, so a demo cannot quietly drift from the
scored runs.

TWO THINGS TO KNOW BEFORE YOU TRUST A SUMMARY.

1. The prospect's real name must appear in the transcript, on the speaker labels. The
   student was trained to refer to the prospect by name and never as "SPEAKER_00", so a
   transcript with anonymised labels gives it nothing to name -- and every summary that
   names the person on the call will then read as invented. This script warns if it sees
   anonymised labels.
2. --model student-blend is the 4-bit file that ships. student-blend-f16 is the 16-bit
   fallback: about 1.85x the wall time, for roughly +0.016 ROUGE-1. Use it if a demo needs
   the best output rather than the shipped artifact.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import student_generate as sg          # noqa: E402  (render through the trained tokenizer)
from measure_local import options, post, OLLAMA  # noqa: E402  (the shipped decoding config)
from transformers import AutoTokenizer  # noqa: E402

DEFAULT_ADAPTER = os.path.join(HERE, "adapters", "student-lora-blend")
INFER_INPUT = os.path.join(HERE, "infer_input.jsonl")
LOG = os.path.join(HERE, "try_calls.jsonl")

STAGES = [
    "Discovery/Qualification",
    "Technical Call Completed",
    "Marketing Agreement Request",
    "Contract Sent",
    "Contract Signed",
    "Lost",
]


def system_prompt():
    """The exact system message the student was trained under, read from the corpus.

    Read rather than retyped: a paraphrase here is a prompt the model never saw, and the
    failure would be invisible in the output.
    """
    with open(INFER_INPUT, encoding="utf-8") as f:
        return json.loads(f.readline())["system"]


def build_row(call_id, transcript, prospect, stage, title=""):
    """The {system, user} shape every training and eval row used.

    Verified against all 144 eval rows: one system prompt, every user field opening with
    a deal stage, every one closing with the bracketed prospect sentence.
    """
    user = "\n".join([
        f"Deal stage: {stage}",
        "",
        "Transcript:",
        "",
        transcript.strip(),
        "",
        f"(The prospect on this call is {prospect}.)",
    ])
    return {"call_id": call_id, "title": title, "system": system_prompt(), "user": user}


REP_LABELS = {"rep", "salesperson", "sales rep", "sales"}


def check_labels(transcript):
    """Warn when the transcript cannot support the grounding the house style asks for.

    Both halves matter and they fail differently. Anonymised labels leave the student with
    nothing to name, so it either avoids naming anyone or invents one. A missing `rep:`
    label means the prompt is off the shape training used -- 143 of the 144 eval rows carry
    it -- which is the class of mismatch this project checks for everywhere else.
    """
    if re.search(r"\bSPEAKER_\d+\b", transcript):
        return ("this transcript uses anonymised SPEAKER_## labels, so the prospect's real "
                "name is not in it. The student has nothing to name and will either avoid "
                "naming anyone or invent a name. Put real names on the labels first.")
    # Case-insensitive: the convention is the literal lowercase `rep:`.
    labels = {m.strip() for m in re.findall(r"(?m)^([^:\n]{1,40}):", transcript)}
    if not any(l.lower() in REP_LABELS for l in labels):
        return ("no 'rep:' label found, and 143 of the 144 eval rows carry one. The house "
                "style calls the salesperson 'the rep', so the prompt is off the shape the "
                "student trained on.")
    return None


def summarize(model, tok, row, num_predict, num_ctx, greedy, keep_alive):
    prompt = sg.render(tok, row)
    t0 = time.time()
    r = post("/api/generate", {
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "keep_alive": keep_alive,
        "options": options(num_predict, num_ctx, greedy=greedy),
    })
    wall = time.time() - t0
    p = r.get("prompt_eval_duration", 0) / 1e9
    d = r.get("eval_duration", 0) / 1e9
    return {
        "call_id": row["call_id"],
        "title": row["title"],
        "model": model,
        "decoding": "greedy (deterministic)" if greedy else "sampled temp 0.7 / top_p 0.9 / rp 1.05",
        "summary": (r.get("response") or "").strip(),
        "prompt_tokens": r.get("prompt_eval_count"),
        "output_tokens": r.get("eval_count"),
        "prefill_s": round(p, 2),
        "decode_s": round(d, 2),
        "wall_s": round(wall, 2),
        "done_reason": r.get("done_reason"),
        "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcript", help="path to a .txt transcript, diarised as 'Name: text'")
    ap.add_argument("--prospect", required=True,
                    help="the counterparty's real name, as it appears on the speaker labels")
    ap.add_argument("--stage", default="Discovery/Qualification",
                    help="deal stage line. Known values: " + " | ".join(STAGES))
    ap.add_argument("--title", default="", help="optional call title, for your own record")
    ap.add_argument("--model", default="student-blend",
                    help="Ollama model; student-blend is the shipped 4-bit file, "
                         "student-blend-f16 the 16-bit fallback")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="tokenizer source; must be the trained chat template")
    ap.add_argument("--num-predict", type=int, default=900)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--greedy", action="store_true",
                    help="decode by argmax. Deterministic, and it cut invented names by a "
                         "third on the 144-call benchmark at the same length. Costs about "
                         "1.16x the wall time. It cannot escape a repetition loop, so it "
                         "carries a small risk of ending mid-sentence")
    ap.add_argument("--keep-alive", default="0",
                    help="0 unloads the model after the call so the timing is a fresh "
                         "call rather than a cached one")
    ap.add_argument("--show-prompt", action="store_true", help="print the rendered prompt")
    args = ap.parse_args()

    with open(args.transcript, encoding="utf-8") as f:
        transcript = f.read()
    if not transcript.strip():
        print("FATAL: the transcript file is empty.", file=sys.stderr)
        raise SystemExit(2)

    warn = check_labels(transcript)
    call_id = os.path.splitext(os.path.basename(args.transcript))[0]
    row = build_row(call_id, transcript, args.prospect, args.stage, args.title)

    tok = AutoTokenizer.from_pretrained(args.adapter)
    print(f"[try] {len(transcript):,} chars | prospect {args.prospect!r} | stage {args.stage!r}")
    print(f"[try] model {args.model} via {OLLAMA}")
    if warn:
        print(f"\n  WARNING: {warn}\n")

    rec = summarize(args.model, tok, row, args.num_predict, args.num_ctx,
                    args.greedy, int(args.keep_alive) if args.keep_alive.isdigit()
                    else args.keep_alive)

    print("=" * 78)
    print(rec["summary"] if rec["summary"] else "(the model returned nothing)")
    print("=" * 78)
    print(f"{rec['output_tokens']} output tokens | prefill {rec['prefill_s']}s + decode "
          f"{rec['decode_s']}s = {rec['wall_s']}s wall | done_reason {rec['done_reason']}")
    if rec["done_reason"] == "length":
        print("NOTE: this hit the 900-token cap and may end mid-sentence. Raise "
              "--num-predict to see whether it terminates on its own.")
    if rec["prompt_tokens"] and rec["prefill_s"]:
        rate = rec["prompt_tokens"] / rec["prefill_s"]
        if rate > 2000:
            print(f"NOTE: prefill implies {rate:,.0f} tok/s, which this CPU cannot do. That "
                  "is a cached prompt prefix, so the timing above is not a fresh-call cost.")

    if args.show_prompt:
        print("\n--- rendered prompt ---")
        print(rec["prompt"])

    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[try] appended the record (summary, timings, prompt) -> {LOG}")


if __name__ == "__main__":
    main()
