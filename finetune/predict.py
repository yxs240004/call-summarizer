"""Generate house-style summaries for the target calls with the fine-tuned model.

Reads infer_input.jsonl (one {call_id, title, system, user} per line), runs the
base Qwen3-8B + our LoRA adapter, and writes results.jsonl.

Runs the same chat template with thinking disabled, so the model answers
directly instead of emitting a <think> block first.
"""
import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def load(args):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Decoder-only models must pad on the LEFT for batched generation, otherwise
    # the pad tokens sit between the prompt and the continuation and the model
    # continues from padding instead of from our text.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return tokenizer, model


def render(tokenizer, row):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="infer_input.jsonl")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--adapter", default=None, help="path to the LoRA adapter dir")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[predict] {len(rows)} prompts | adapter={args.adapter}")

    tokenizer, model = load(args)

    out = open(args.out, "w", encoding="utf-8")
    done = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [render(tokenizer, r) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
            model.device
        )
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
        # Slice off the prompt; only the continuation is the summary.
        for row, ids in zip(batch, gen):
            text = tokenizer.decode(ids[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            out.write(json.dumps({"call_id": row["call_id"], "title": row["title"], "summary": text},
                                 ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            print(f"[{done}/{len(rows)}] {row['call_id']} ({dt / len(batch):.1f}s) {text[:90]}...")
    out.close()
    print(f"[predict] wrote {done} -> {args.out}")


if __name__ == "__main__":
    main()
