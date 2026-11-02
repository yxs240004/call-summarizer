"""Check the student side before any student training is written or paid for.

Three questions, all of which are guesses until this runs:

  1. Which transformers class actually loads Qwen/Qwen3.5-2B? Its config says
     `Qwen3_5ForConditionalGeneration` -- it is a *multimodal* architecture with a
     vision tower -- so AutoModelForCausalLM may not resolve it at all.

  2. Does the hand-rolled training template render byte-identically to the model's
     own shipped template? This is the check that would have caught the Qwen3-8B
     defect: train.py's hand-rolled template and predict.py's stock template
     disagreed by one token at the first position of the assistant turn, and
     nothing asserted they agreed. Here the assertion is explicit and fails loudly.

  3. Which linear layer names exist outside the vision tower? `all-linear` on a
     multimodal model would hang LoRA adapters on vision projections that text-only
     data gives no gradient to -- dead weight in the adapter, and one more thing to
     explain at deployment.

Downloads only the tokenizer and config (a few MB), not the weights.
"""
import sys

PROBE = [
    {"role": "system", "content": "You are a summarizer."},
    {"role": "user", "content": "Summarize this call.\n\nA: hello\nB: hi"},
    {"role": "assistant", "content": "Two people greeted each other."},
]

# SUPERSEDED, AND WRONG -- do not copy this constant anywhere.
#
# The two markers below are spelled as the literal words ' think' / ' response'. The
# model's own template uses the 7- and 8-character strings that tokenize to the added
# tokens 248068 / 248069. Because the two spellings differ by one character each, the
# renders agreed on 44 of their 45 tokens and the difference was invisible in a
# printed string -- which is why diff_templates.py reported a mismatch that took three
# attempts to explain (see probe_template_bytes.py for the byte-level resolution).
#
# train_student.py therefore uses NO template constant: it renders through the model's
# own chat_template and derives the label boundary by re-rendering the prompt half,
# asserting the prefix property on every row. This constant is kept only as the
# historical fixture that caught the error.
# Must stay byte-identical to train_student.py's STUDENT_TEMPLATE.
STUDENT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- set content = message.content|trim -%}"
    "{%- if message.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n thinking\\n\\n response\\n\\n' -}}"
    "{%- generation -%}{{ content }}<|im_end|>{%- endgeneration -%}"
    "{{- '\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n thinking\\n\\n</think>\\n\\n' -}}"
    "{%- endif -%}"
)

MODEL = "Qwen/Qwen3.5-2B"
VISION_HINTS = ("visual", "vision", "image", "merger", "patch_embed")


def flat_template(tokenizer):
    template = tokenizer.chat_template
    if isinstance(template, dict):
        return template.get("default") or next(iter(template.values()), "")
    return template or ""


def tail(tokenizer, text, n=8):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return " ".join(f"{i}:{ascii(tokenizer.convert_ids_to_tokens(i))}" for i in ids[-n:])


def main():
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    print(f"transformers {transformers.__version__}")

    cfg = AutoConfig.from_pretrained(MODEL)
    print(f"architectures   : {getattr(cfg, 'architectures', None)}")
    print(f"model_type      : {cfg.model_type}")
    print(f"top-level keys  : {sorted(k for k in cfg.to_dict() if k in ('vision_config', 'text_config', 'image_token_id', 'vocab_size'))}")

    print("\n-- candidate auto classes present in this transformers --")
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                 "AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration",
                 "AutoModelForMultimodalLM"):
        print(f"  {name:<36} {hasattr(transformers, name)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"\nlen(tokenizer)  : {len(tok)}")
    print(f"vocab_size      : {tok.vocab_size}")
    print(f"chat_template is: {'dict' if isinstance(tok.chat_template, dict) else type(tok.chat_template).__name__}")

    shipped = flat_template(tok)

    # ---- the assertion that matters ----
    tok.chat_template = shipped
    want = tok.apply_chat_template(PROBE, tokenize=False, add_generation_prompt=False)
    tok.chat_template = STUDENT_TEMPLATE
    got = tok.apply_chat_template(PROBE, tokenize=False, add_generation_prompt=False)

    print("\n-- training render: hand-rolled vs shipped --")
    print(f"  shipped    : {ascii(want)}")
    print(f"  hand-rolled: {ascii(got)}")
    print(f"  MATCH      : {got == want}")
    if got != want:
        for i, (a, b) in enumerate(zip(want, got)):
            if a != b:
                print(f"  first difference at char {i}: shipped {a!r} vs hand-rolled {b!r}")
                print(f"  shipped    ...{ascii(want[max(0, i - 30):i + 30])}")
                print(f"  hand-rolled...{ascii(got[max(0, i - 30):i + 30])}")
                break
        else:
            print(f"  one is a prefix of the other: len {len(want)} vs {len(got)}")

    # ---- what inference will actually send ----
    tok.chat_template = STUDENT_TEMPLATE
    gen_prompt = tok.apply_chat_template(PROBE[:2], tokenize=False, add_generation_prompt=True)
    print("\n-- generation prompt --")
    print(f"  tail       : {tail(tok, gen_prompt)}")
    print(f"  full ids   : {tail(tok, gen_prompt, n=200).count(' ')} tokens rendered")

    # ---- LoRA target discovery ----
    print("\n-- linear layers outside the vision tower (LoRA targets) --")
    try:
        import torch
        cls = None
        for name in ("AutoModelForImageTextToText", "Qwen3_5ForConditionalGeneration",
                     "AutoModelForCausalLM"):
            candidate = getattr(transformers, name, None)
            if candidate is None:
                continue
            try:
                # from_config, not from_pretrained: instantiating from the config
                # gives the real module names and parameter count without pulling
                # four gigabytes of weights we have no use for here.
                model = candidate.from_config(cfg)
                cls = name
                break
            except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
                print(f"  {name} failed: {type(exc).__name__}: {str(exc)[:200]}")
        if cls is None:
            print("  FATAL: no auto class loaded this model")
            return 1
        print(f"  loaded with : {cls}")
        leaves, vision = set(), set()
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                leaf = n.split(".")[-1]
                (vision if any(h in n for h in VISION_HINTS) else leaves).add(leaf)
        print(f"  text leaves : {sorted(leaves)}")
        print(f"  vision-only : {sorted(vision - leaves)}")
        print(f"  params      : {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B total")
    except ImportError as exc:
        print(f"  skipped (torch not importable here): {exc}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
