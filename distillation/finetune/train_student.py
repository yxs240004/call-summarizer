"""Train the distillation student (Qwen3.5-2B) on the teacher's summaries.

Step 4 of DESIGN.md. Two arms, one recipe: `blend` trains on the teacher's
pseudo-labels, `control` on the original human references. Same row count, same
hyperparameters, same seed -- so the only thing that differs is what the assistant
turn says.

Where the labels come from, and why there is no hand-rolled template here
-----------------------------------------------------------------------
A hand-rolled chat template has silently disagreed with the one the model actually
ships twice in this project:

  * Qwen3-8B  -- train.py's QWEN3_TEMPLATE emitted the empty thinking block as
                 ordinary text (token 7274) where predict.py's stock template emitted
                 the special token 151667. One token apart, at the first position of
                 the assistant turn, which is the position that tells a Qwen3 model
                 whether it is in reasoning mode.
  * Qwen3.5-2B -- the first transcription of the shipped template read its markers as
                 the words ' think' / ' response' (tokens 7047 / 1965). The model
                 actually uses the added tokens 248068 / 248069. The two renders
                 agreed on 44 of their 45 tokens.

Both were invisible in a printed string and obvious in a token id. So this file has no
template constant. It renders through the model's own chat_template and finds the
assistant span by re-rendering the prompt half:

    full   = apply_chat_template(messages,      add_generation_prompt=False)
    prompt = apply_chat_template(messages[:-1], add_generation_prompt=True)

and then asserts `prompt == full[:len(prompt)]` on every row. While that holds, the
boundary is exact and nothing about the prompt shape has been assumed; when it does
not, the row is rejected loudly rather than quietly trained on. It is the same idea as
the teacher pass's template assertion, applied to the student.

That is also why there are no `{% generation %}` markers and no
`assistant_only_loss` flag: the mask is computed from the ids themselves, so there is
no template to patch and nothing to transcribe.

Recipe notes
------------
  * LoRA targets are the text-side linear leaves only. Qwen3.5-2B is multimodal
    (Qwen3_5ForConditionalGeneration: a vision tower plus a text stack), so
    `all-linear` would hang adapters on vision projections that text-only data gives
    no gradient to -- adapter weights that do nothing, and one more thing to explain
    at deployment.
  * lm_head is left out. The student ties word embeddings, and a fresh low-rank head
    on a tied weight is the one module where that interaction can bite.
  * No gradient checkpointing: rows run about 1.5k tokens at batch size 1, so
    activations are small and the card has room. Leaving it off removes a real failure
    mode (checkpointing + PEFT + a multimodal wrapper) at no cost.
  * No 4-bit quantization. 2B in bf16 fits comfortably, and quantizing the thing we
    are about to quantize again would only confuse the evaluation.
  * Everything else (r=16, lora_alpha=32, dropout 0.05, lr 2e-4 cosine, warmup 5,
    3 epochs, gradient accumulation 8) matches train.py, so the two student arms stay
    comparable with each other and with the teacher's own fine-tune.
"""
import argparse
import json
import math
import os
import random
import time
from collections.abc import Mapping

MODEL = "Qwen/Qwen3.5-2B"

# Text-side linear leaves on Qwen3.5-2B, taken from verify_student.py's module walk.
# The vision tower's leaves (linear_fc1, linear_fc2, proj, qkv) are deliberately absent.
TEXT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
]
VISION_HINTS = ("visual", "vision", "image", "merger", "patch_embed")

# Built from code points rather than written out: a literal marker in this file is
# exactly the thing that got transcribed wrong once already.
LT, GT = chr(0x3C), chr(0x3E)
MARK_OPEN = LT + "think" + GT
MARK_CLOSE = LT + "/think" + GT


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_id_list(rendered):
    """apply_chat_template(tokenize=True) does not hand back a flat list of ints.

    In transformers 5.x it returns a BatchEncoding, which is NOT a dict subclass, so an
    isinstance(..., dict) test falls through and iterating it yields its *keys*
    ('input_ids', 'attention_mask') rather than tokens. That is exactly how this
    function first failed: both renders collapsed to those same two strings, the prefix
    assertion passed vacuously (a list of 2 equals itself), and the guard surfaced it as
    'the assistant turn contributed no tokens'. So test for a mapping properly and take
    input_ids by name, then unwind whatever nesting the version used.
    """
    if isinstance(rendered, Mapping) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    rendered = list(rendered)
    if not rendered:
        return []
    first = rendered[0]
    if hasattr(first, "ids"):                     # per-sample Encoding object
        return [int(i) for i in first.ids]
    if isinstance(first, (list, tuple)):          # a batch of one row
        return [int(i) for i in first]
    return [int(i) for i in rendered]


def flat_template(tokenizer):
    template = tokenizer.chat_template
    if isinstance(template, dict):
        return template.get("default") or next(iter(template.values()), "")
    return template or ""


def report_markers(tokenizer):
    """Record what the shipped template actually uses for the empty thinking block.

    Printed as codepoints and token ids -- a printed string already hid this error
    twice, so the log gets the ordinals.
    """
    template = flat_template(tokenizer)
    print(f"[marker] template: {len(template)} chars, {type(tokenizer.chat_template).__name__}")
    ok = True
    for name, mark in (("open", MARK_OPEN), ("close", MARK_CLOSE)):
        ids = tokenizer(mark, add_special_tokens=False)["input_ids"]
        points = " ".join(str(ord(c)) for c in mark)
        present = mark in template
        ok = ok and present
        print(f"[marker] {name}: codepoints [{points}] -> ids {ids}  in template: {present}")
    if not ok:
        raise SystemExit("[marker] FATAL: the shipped template does not use these markers; "
                         "the assumptions in this file are wrong -- stopping before training")
    return template


def encode_pair(tokenizer, messages, max_length):
    """(input_ids, labels, boundary) for one example whose last turn is the assistant's.

    The boundary is the number of leading ids that belong to the prompt, and it is
    verified rather than assumed.
    """
    full = as_id_list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False))
    prompt = as_id_list(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True))

    if full[: len(prompt)] != prompt:
        diverge = next((k for k, (a, b) in enumerate(zip(full, prompt)) if a != b),
                       min(len(full), len(prompt)))
        raise ValueError(
            f"prompt half is not a prefix of the full render: first divergence at token "
            f"{diverge} (full {full[diverge:diverge + 3]} vs prompt {prompt[diverge:diverge + 3]}), "
            f"len(full)={len(full)} len(prompt)={len(prompt)}")
    if len(full) > max_length:
        raise ValueError(f"row is {len(full)} tokens, over --max-length {max_length}; "
                         "refusing to truncate silently")
    labels = list(full)
    if len(labels) == len(prompt):
        raise ValueError("the assistant turn contributed no tokens -- nothing to learn from")
    for i in range(len(prompt)):
        labels[i] = -100
    return full, labels, len(prompt)


class Examples:
    """Pre-tokenized rows. Built eagerly so a contract violation fails before the GPU
    is touched, not on step 40."""

    def __init__(self, path, tokenizer, max_length):
        rows = load_jsonl(path)
        if not rows:
            raise SystemExit(f"FATAL: no rows in {path}")
        self.pad_id = tokenizer.pad_token_id
        self.examples = []
        for row in rows:
            ids, labels, boundary = encode_pair(tokenizer, row["messages"], max_length)
            if not any(l != -100 for l in labels):
                raise SystemExit(f"FATAL: {row.get('call_id')} supervises nothing")
            self.examples.append({
                "input_ids": ids,
                "labels": labels,
                "boundary": boundary,
                "call_id": row.get("call_id", ""),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def collate(self, batch):
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "boundary": [b["boundary"] for b in batch],
        }

    def describe(self):
        lens = sorted(len(e["input_ids"]) for e in self.examples)
        sup = sorted(sum(1 for l in e["labels"] if l != -100) for e in self.examples)
        bounds = sorted(e["boundary"] for e in self.examples)
        print(f"[data] {len(self.examples)} rows")
        print(f"[data] total tokens  min {lens[0]} median {lens[len(lens) // 2]} max {lens[-1]}")
        print(f"[data] supervised    min {sup[0]} median {sup[len(sup) // 2]} max {sup[-1]}")
        print(f"[data] prompt boundary min {bounds[0]} median {bounds[len(bounds) // 2]} max {bounds[-1]}")


def load_model(model_name):
    import torch
    from transformers import AutoModelForImageTextToText

    # AutoModelForImageTextToText, not AutoModelForCausalLM: Qwen3.5-2B is
    # Qwen3_5ForConditionalGeneration, a multimodal class. No images are ever passed;
    # the vision tower just sits there with requires_grad=False.
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16)

    # A multimodal config does not carry pad_token_id/eos_token_id at the top level --
    # they live on text_config, and reading the top-level name raises AttributeError
    # rather than returning None. Training only needs the *tokenizer's* pad id, but a
    # missing entry here breaks generate() later, so fill it in on whichever configs
    # exist, without assuming either one is present.
    config = model.config
    levels = [c for c in (getattr(config, "text_config", None), config) if c is not None]
    eos = next((getattr(c, "eos_token_id", None) for c in levels
                if getattr(c, "eos_token_id", None) is not None), None)
    if eos is not None:
        for c in levels:
            if getattr(c, "pad_token_id", None) is None:
                setattr(c, "pad_token_id", eos)
    print(f"[model] pad_token_id={getattr(config, 'pad_token_id', None)} "
          f"text_config.eos_token_id={getattr(levels[0], 'eos_token_id', None)}")
    return model


def attach_lora(model):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=TEXT_TARGETS, task_type="CAUSAL_LM",
    )
    try:
        pefted = get_peft_model(model, config)
    except Exception as exc:  # noqa: BLE001 - PEFT's task-type checks are version-specific
        print(f"[lora] task_type=CAUSAL_LM rejected ({type(exc).__name__}: {exc}); retrying unset")
        config.task_type = None
        pefted = get_peft_model(model, config)

    # The base runs bf16; the adapter stays fp32 so the optimizer keeps real precision.
    for _, param in pefted.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    vision_trainable = [n for n, p in pefted.named_parameters()
                        if p.requires_grad and any(h in n.lower() for h in VISION_HINTS)]
    if vision_trainable:
        raise SystemExit(f"[lora] FATAL: vision parameters are trainable: {vision_trainable[:5]}")
    pefted.print_trainable_parameters()
    return pefted


def evaluate(model, data, device):
    import torch

    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data), 2):
            batch = data.collate([data[j] for j in range(i, min(i + 2, len(data)))])
            batch.pop("boundary")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total += float(model(**batch).loss)
            batches += 1
    model.train()
    return total / max(batches, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", required=True, help="student_data/<arm>_train.jsonl")
    ap.add_argument("--val", default="", help="optional; student_data/val.jsonl")
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="truncate the training set (smoke runs)")
    ap.add_argument("--verify", action="store_true",
                    help="report the tokenization contract and exit without loading weights")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report_markers(tokenizer)

    train = Examples(args.data, tokenizer, args.max_length)
    if args.limit:
        train.examples = train.examples[: args.limit]
    train.describe()

    val = Examples(args.val, tokenizer, args.max_length) if args.val else None
    if val:
        print(f"[data] val: {len(val)} rows")

    print("[contract] every training row: prompt half is an exact prefix of the full render")
    if args.verify:
        # Show one row end to end so the log records the prompt/target split itself.
        sample = train[0]
        boundary = sample["boundary"]
        ids = sample["input_ids"]
        print(f"[contract] row 0 call_id={sample['call_id']}")
        print(f"[contract]   prompt tail ids {ids[boundary - 4:boundary]}")
        print(f"[contract]   decoded prompt tail {tokenizer.decode(ids[boundary - 4:boundary])!r}")
        print(f"[contract]   target head       {tokenizer.decode([i for i in ids[boundary:] if i != -100][:24])!r}")
        print("[verify] tokenization contract holds; nothing loaded, nothing written")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("FATAL: no GPU visible; this is meant to run on the pod")

    model = attach_lora(load_model(args.model)).to(device)
    model.config.use_cache = False
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    # Cosine with a warmup, the same shape train.py used for the teacher, so the two
    # arms differ only in their data and stay comparable with the 8B's own fine-tune.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup, num_training_steps=total_steps,
    )
    print(f"[train] {len(train)} rows, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    order = list(range(len(train)))
    history, t_start, step = [], time.time(), 0
    for epoch in range(args.epochs):
        random.shuffle(order)
        running, seen = 0.0, 0
        for i in range(0, len(order), args.grad_accum):
            chunk = order[i: i + args.grad_accum]
            optimizer.zero_grad(set_to_none=True)
            for j in chunk:
                batch = train.collate([train[j]])
                batch.pop("boundary")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise SystemExit(f"[train] FATAL: non-finite loss at step {step}: {loss.detach().item()}")
                (loss / len(chunk)).backward()
                running += loss.detach().item()
                seen += 1
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"[train] epoch {epoch + 1} step {step}/{total_steps} "
                      f"loss {running / max(seen, 1):.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed {(time.time() - t_start) / 60:.1f}m")
                running, seen = 0.0, 0
        record = {"epoch": epoch + 1, "steps": step}
        if val:
            record["val_loss"] = evaluate(model, val, device)
            print(f"[train] epoch {epoch + 1} val_loss {record['val_loss']:.4f}")
        history.append(record)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    # The tokenizer goes with the adapter, so whatever loads this directory prompts it
    # the way its own template says -- the mistake teacher_generate.py exists to avoid.
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data": os.path.abspath(args.data),
            "rows": len(train),
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "target_modules": TEXT_TARGETS,
            "history": history,
            "seconds": round(time.time() - t_start, 1),
        }, f, indent=2)
    print(f"[train] saved adapter -> {args.out} in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
