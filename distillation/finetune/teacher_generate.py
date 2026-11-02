"""Generate the distillation pseudo-labels with the fine-tuned teacher.

Step 2 of the distillation build (see DESIGN.md). Reads teacher_input.jsonl -- the
154 holdout prompts written by make_teacher_input.py -- and writes one teacher
summary per call to teacher_output.jsonl.

Why this is not predict.py with different flags
-----------------------------------------------
predict.py renders through the *base* Qwen3 tokenizer's stock chat template with
enable_thinking=False, which emits the empty thinking block as the single special
token 151667. train.py rendered that same block through a hand-rolled template that
spells it as ordinary text, which tokenizes to 7274 instead. The two are one token
apart, at the first position of the assistant turn -- the position that tells a
Qwen3 model whether it is in reasoning mode. So results.jsonl was generated from a
prompt shape the 8B was never trained on.

This script loads the tokenizer from the adapter directory instead. train.py wrote
that directory with tokenizer.save_pretrained(OUT_DIR) immediately after assigning
its own template, so its chat_template.jinja IS the training template, byte for
byte, and the teacher gets prompted the way it was trained. The startup check
asserts that against train.py's own constant whenever train.py is alongside, and
prints the tokenized tail of a real generation prompt either way, so a mismatch
shows up in the log rather than in the outputs.

No enable_thinking kwarg is passed anywhere: the hand-rolled template already
hard-codes the empty thinking block, so there is nothing for the kwarg to do.

Decoding follows DESIGN.md D7: beam search, one pseudo-label per call. Sampled
deliverables and teaching targets are different jobs -- the beam is the teacher's
highest-probability sequence, which is the cleaner thing to imitate. The repetition
penalty is off for the same reason: it would distort the sequence being asked for.

The run is resumable. --resume drops calls already present in the output file, so
an interrupted GPU hour is not paid for twice.
"""
import argparse
import ast
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER = os.path.join(HERE, "qwen3-8b-lora")
DEFAULT_INPUT = os.path.join(HERE, "teacher_input.jsonl")
DEFAULT_OUT = os.path.join(HERE, "teacher_output.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def training_template(path=os.path.join(HERE, "train.py")):
    """Recover train.py's QWEN3_TEMPLATE from source, or None if train.py is absent.

    Read rather than imported: train.py is a training entry point and running it
    would start a training job.
    """
    if not os.path.exists(path):
        return None
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "QWEN3_TEMPLATE" for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def flat_template(tokenizer):
    """The tokenizer's chat template as one string.

    transformers 5.x stores templates as a mapping keyed by name ({"default": ...})
    where 4.x stored a bare string, so read either shape rather than crashing on
    the version the pod happens to have installed.
    """
    template = tokenizer.chat_template
    if isinstance(template, dict):
        return template.get("default") or next(iter(template.values()), "")
    return template or ""


def load(adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # From the adapter dir, not BASE_MODEL: this is the whole point of the script.
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    # Decoder-only models pad on the LEFT for batched generation, or the pad tokens
    # sit between the prompt and the continuation and the model continues from them.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `dtype`, not `torch_dtype`: transformers 5.x renamed the argument, and the
    # old spelling no longer does anything (silently ignored at best).
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def render(tokenizer, row):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def describe_generation_prompt(tokenizer, prompt):
    """The last few tokens of the rendered prompt, so the log records the shape the
    teacher actually saw. This is where a template mismatch becomes visible."""
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tail = ids[-6:]
    shown = " ".join(f"{i}:{ascii(tokenizer.convert_ids_to_tokens(i))}" for i in tail)
    return shown


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="prompts per forward pass; beam count multiplies this in memory")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = off; anything else distorts the sequence beam search is optimising")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip calls already present in --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and report prompt shapes without loading the model")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    # ---- template check: is the adapter's saved template the one training used? ----
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    source = training_template()
    saved = flat_template(tokenizer)
    if source is None:
        print("[template] train.py not alongside; cannot assert against training")
    else:
        same = source.strip() == saved.strip()
        print(f"[template] adapter template == train.py's QWEN3_TEMPLATE: {same}")
        if not same:
            print("[template] FATAL: the adapter's template is not the one training used. "
                  "Every pseudo-label would be off-distribution -- stopping.")
            raise SystemExit(2)

    sample = render(tokenizer, rows[0])
    print(f"[template] generation prompt tail -> {describe_generation_prompt(tokenizer, sample)}")
    print("[template] (id 151667 is the  thinking SPECIAL token; 7274 is the text ' thinking')")

    if args.dry_run:
        lens = []
        for r in rows:
            lens.append(len(tokenizer(render(tokenizer, r), add_special_tokens=False)["input_ids"]))
        lens.sort()
        print(f"[dry-run] {len(lens)} prompts, tokens: "
              f"min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}")
        print("[dry-run] no model loaded, nothing written")
        return

    # ---- resume ----
    done = {}
    if args.resume and os.path.exists(args.out):
        done = {r["call_id"]: r for r in load_jsonl(args.out)}
        rows = [r for r in rows if r["call_id"] not in done]
        print(f"[resume] {len(done)} calls already in {os.path.basename(args.out)}, {len(rows)} left to do")

    if not rows:
        print("[teacher] nothing to do")
        return

    import torch

    tokenizer, model = load(args.adapter)
    out = open(args.out, "a" if done else "w", encoding="utf-8")

    n_new, empties, t_start = 0, [], time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        prompts = [render(tokenizer, r) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_length).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.time() - t0
        prompt_len = enc["input_ids"].shape[1]
        # Slice off the prompt; only the continuation is the summary.
        for row, ids in zip(batch, gen):
            new_ids = ids[prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            rec = {
                "call_id": row["call_id"],
                "split": row["split"],
                "title": row["title"],
                "summary": text,
                "summary_tokens": int(len(new_ids)),
                "seconds": round(dt / len(batch), 2),
                "decode": "beam" + str(args.num_beams),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_new += 1
            if not text:
                empties.append(row["call_id"])
            print(f"[{n_new}/{len(rows)}] {row['call_id']} "
                  f"{rec['summary_tokens']}tok {rec['seconds']}s {text[:80]!r}")
    out.close()

    total = time.time() - t_start
    print(f"\n[teacher] wrote {n_new} -> {args.out} in {total / 60:.1f} min "
          f"({total / max(n_new, 1):.1f}s per call)")
    if empties:
        print(f"[teacher] WARNING: {len(empties)} empty summaries: {empties[:10]}")
        print("[teacher] an empty pseudo-label teaches the student to say nothing — inspect before training")


if __name__ == "__main__":
    main()
