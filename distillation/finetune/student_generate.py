"""Generate the student's summaries for the 144 benchmark calls.

Step 6 of the distillation build (see DESIGN.md D8). Loads Qwen3.5-2B plus one arm's
LoRA adapter and writes one summary per call, in the same shape as results.jsonl, so
the student can be scored against `qwen4b` 0.4410 and `qwen8b_honest` 0.4372 on the
same 144 calls.

The tokenizer comes from the adapter directory, not from the base model, and decoding
settings mirror predict.py exactly (sampling, temperature 0.7, top_p 0.9, repetition
penalty 1.05). Both matter for comparability: the prompt must be the shape the student
trained on, and the decoding must be the shape the baselines were measured with, or a
score difference could be either one.

The startup check is the student-side twin of teacher_generate.py's. It renders a
generation prompt and asserts that it ends with the same marker block the training
renders ended with, derived from the tokenizer rather than hardcoded. A chat-template
mismatch between training and inference is the defect this project has already hit
twice; here it is checked in the direction that actually ships.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3.5-2B"

LT, GT = chr(0x3C), chr(0x3E)
MARK_OPEN = LT + "think" + GT
MARK_CLOSE = LT + "/think" + GT


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # AutoModelForImageTextToText: the student is multimodal. No images are ever passed.
    # `dtype`, not `torch_dtype` -- transformers 5.x renamed the argument, and predict.py
    # still uses the old spelling, which is silently ignored and leaves the model fp32.
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def device_of(model):
    """Where the inputs must go.

    Not model.device: with device_map="auto" the model is dispatched across devices, and
    the attribute resolves inconsistently (and on a multimodal wrapper may not resolve at
    all). The embedding's device is the one the input ids have to match, so read it off
    the first parameter instead.
    """
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def render(tokenizer, row):
    """The generation prompt. No enable_thinking kwarg: the student's own template reads
    that variable and falls through to the empty thinking block when it is undefined,
    which is the branch training's renders took."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def marker_tail(tokenizer):
    """The ids a training prompt ended with, derived rather than hardcoded."""
    text = MARK_OPEN + "\n\n" + MARK_CLOSE + "\n\n"
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=os.path.join(HERE, "infer_input.jsonl"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[student] {len(rows)} prompts | adapter={args.adapter}")

    import torch
    from transformers import AutoTokenizer

    # Check the template before the weights: cheap, and it is the failure that hides.
    probe_tok = AutoTokenizer.from_pretrained(args.adapter)
    tail = marker_tail(probe_tok)
    sample_ids = probe_tok(render(probe_tok, rows[0]), add_special_tokens=False)["input_ids"]
    got = sample_ids[-len(tail):]
    print(f"[template] expected marker tail {tail} -> {probe_tok.decode(got)!r}")
    print(f"[template] generation prompt ends with the training block: {got == tail}")
    if got != tail:
        print("[template] FATAL: the generation prompt does not end with the block training "
              "used. Every sample would be off-distribution -- stopping.")
        raise SystemExit(2)

    tokenizer, model = load(args.adapter)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(args.out, "w", encoding="utf-8")
    done, t_start = 0, time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        prompts = [render(tokenizer, r) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_length).to(device_of(model))
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.time() - t0
        prompt_len = enc["input_ids"].shape[1]
        for row, ids in zip(batch, gen):
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
            out.write(json.dumps({
                "call_id": row["call_id"],
                "title": row.get("title", ""),
                "summary": text,
                "summary_tokens": int(len(ids[prompt_len:])),
                "seconds": round(dt / len(batch), 2),
                "decode": "sample-t0.7-p0.9",
            }, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            print(f"[{done}/{len(rows)}] {row['call_id']} {dt / len(batch):.1f}s {text[:80]!r}")
    out.close()

    total = time.time() - t_start
    print(f"\n[student] wrote {done} -> {args.out} in {total / 60:.1f} min "
          f"({total / max(done, 1):.1f}s per call)")


if __name__ == "__main__":
    main()
