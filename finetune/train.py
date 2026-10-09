"""LoRA fine-tune Qwen3-8B on our (transcript + metadata) -> house-style pairs.

Runs on a single 24GB GPU. Reads train.jsonl / val.jsonl (chat messages) and
writes a LoRA adapter to ./qwen3-8b-lora.

Notes:
  - Qwen3 is a "hybrid thinking" model. We disable thinking mode
    (enable_thinking=False) so it writes the summary directly, matching the
    training targets (which contain no chain-of-thought).
  - assistant_only_loss=True masks the loss to the assistant turn, so the model
    is graded only on the summary it produces, not on re-predicting the input.
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "qwen3-8b-lora"))

# Qwen3's shipped chat template has two problems for us on TRL 1.x:
#   1. It defaults to THINKING ON when `enable_thinking` is undefined, and the
#      trainer no longer forwards chat_template_kwargs, so we can't turn it off.
#   2. It has no {% generation %} markers, so `assistant_only_loss` can't build
#      its loss mask.
# Our data is always exactly [system, user, assistant], so we define the template
# ourselves: hardcode the empty think block and bracket the assistant turn.
QWEN3_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{{- ' thinking\\n\\n</think>\\n\\n' -}}"
    "{%- generation -%}{{ message.content }}<|im_end|>{%- endgeneration -%}"
    "{{- '\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{{- ' thinking\\n\\n</think>\\n\\n' -}}"
    "{%- endif -%}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the config")
    ap.add_argument("--epochs", type=float, default=3)
    args = ap.parse_args()

    train_ds = load_dataset("json", data_files=os.path.join(HERE, "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=os.path.join(HERE, "val.jsonl"), split="train")
    if args.smoke:
        train_ds, val_ds = train_ds.select(range(min(8, len(train_ds)))), val_ds.select(range(2))
    print(f"train examples: {len(train_ds)} | val examples: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = QWEN3_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    # Sanity check: the rendered training text must line up with what inference
    # will send, or the model learns a format it never sees in production.
    if args.smoke:
        print("--- rendered (tail) ---")
        print(repr(tokenizer.apply_chat_template(
            train_ds[0]["messages"], tokenize=False, add_generation_prompt=False)[-160:]))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args_cfg = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        # TRL 1.x dropped warmup_ratio in favour of an absolute step count.
        # ~100 total steps, so ~5 warmup steps is about the same 3-5%.
        warmup_steps=5,
        logging_steps=1 if args.smoke else 5,
        save_strategy="no" if args.smoke else "epoch",
        eval_strategy="no" if args.smoke else "epoch",
        bf16=True,
        # Recompute activations in the backward pass instead of storing them.
        # Costs ~30% more compute for a large drop in activation memory, which is
        # what makes an 8B model + 4096-token examples fit on a 24GB card.
        gradient_checkpointing=True,
        max_length=4096,
        packing=False,
        assistant_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    if not args.smoke:
        trainer.save_model(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print("saved LoRA adapter ->", OUT_DIR)
    else:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
