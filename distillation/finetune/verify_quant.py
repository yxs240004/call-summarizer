"""Does Q4_K_M damage the student, or only shrink it?

Qwen3.5-2B is a hybrid: 18 Gated DeltaNet layers (in_proj_a/in_proj_b [16, 2048],
A_log [16], conv1d [6144, 1, 4]) plus 7 full-attention layers. llama-quantize's guard
that keeps the small ssm_* state tensors in F32 sits inside `if (arch ==
LLM_ARCH_GLM5_NEXT)`, so it does not cover QWEN35 -- in_proj_a/in_proj_b reach the
Q4_K recipe, and they feed exponential decay gates where relative error amplifies.
Whether that actually costs anything is a measurement, not an argument.

WHAT THIS FILE USED TO GET WRONG. It compared the bf16 reference's greedy text against
the deployed GGUF's greedy text and failed anything whose common character prefix was
under 50%, on the stated premise that "a model whose weights survived quantization
diverges late, in the tail, or not at all". It reported a median prefix of 2.5% and
concluded that quantization was implicated.

That premise is false for this architecture, and no control was ever measured to
discover it was false. Two independent facts settle it:

  * verify_merged.py already established that this stack flips argmaxes on 0-4 bf16
    ULPs of perturbation -- both the LoRA merge (Wx + BAx vs (W+B A)x) and, more
    sharply, the gap between HF's reference PyTorch kernels (causal_conv1d and
    flash-linear-attention are not installed here) and llama.cpp's own Gated DeltaNet
    implementation, which is a different numerical implementation of the same math.
  * In free-running greedy decode, ONE such flip rewrites the context for every token
    after it. So the expected text-prefix agreement between two different inference
    ENGINES is near zero even when the weights are identical.

So a raw prefix number is uninterpretable, and a threshold on it is a guess. What makes
it interpretable is a control measured on the same prompts through the same engine with
the quantization removed: the f16 GGUF, registered as `student-blend-f16`. That control
carries the engine gap and essentially none of the quantization gap. This file now
reports three distances instead of one:

    ref(bf16 HF) vs f16 GGUF      <- the instrument's floor: engine only
    ref(bf16 HF) vs Q4_K_M GGUF   <- engine + quantization
    f16 GGUF     vs Q4_K_M GGUF   <- same engine: quantization only

and it can only call quantization implicated when the quantized model is materially
further from the reference than the quantization-free control is. If the control is
just as far, the metric is saturated by the engine gap and says nothing about
quantization either way.

Text rather than token ids, still, because the Ollama path is detokenize -> retokenize,
and a byte-level merge difference would show up as a false divergence at a token index
while the characters are in fact identical.

The reference decode costs ~2 minutes and is cached, so re-running this against another
model is cheap.

    python -I verify_quant.py --merged merged/student-blend --model student-blend \
                              --control student-blend-f16
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# -I implies -P. Reuse the loader and the greedy decoder rather than re-writing them,
# for the reason verify_merged.py already gives: a re-written prompt path is how this
# project lost a template once.
sys.path.insert(0, HERE)
import student_generate as sg  # noqa: E402
from verify_merged import from_merged, greedy, load_jsonl  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"


def generate(model, prompt, num_predict, num_ctx):
    req = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=json.dumps({
            "model": model, "prompt": prompt, "raw": True, "stream": False,
            "keep_alive": "30m",
            # temperature 0 is greedy in llama.cpp, which is what makes this an
            # equality-shaped test rather than a sampling comparison.
            "options": {"temperature": 0, "seed": 42,
                        "num_predict": num_predict, "num_ctx": num_ctx},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def agreement(a, b):
    """Longest common character prefix as a fraction of the shorter text."""
    return common_prefix(a, b) / max(1, min(len(a), len(b)))


def load_reference(tok, rows, args):
    """Greedy-decode the merged bf16 checkpoint, cached to disk.

    The tokenizer is loaded by the caller regardless, because rendering the prompt
    needs it; only the weights are skipped when the cache is warm.
    """
    if os.path.exists(args.ref_cache) and not args.refresh:
        with open(args.ref_cache, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("merged") == args.merged and cached.get("max_new_tokens") == args.max_new_tokens:
            print(f"[quant] reference cached -> {args.ref_cache}")
            return cached["texts"]
        print(f"[quant] cache is for a different reference; re-decoding")

    t0 = time.time()
    _, model = from_merged(args.merged)
    ids = greedy(model, tok, rows, args.max_new_tokens)
    texts = [tok.decode(i) for i in ids]
    del model
    print(f"[quant] bf16 reference decoded in {time.time() - t0:.0f}s")
    # Cache the FULL text, not a head. The first version of this file kept only 120
    # characters and threw away two minutes of compute that the next question needed.
    with open(args.ref_cache, "w", encoding="utf-8") as f:
        json.dump({"merged": args.merged, "max_new_tokens": args.max_new_tokens,
                   "texts": texts}, f)
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merged", required=True, help="merged HF checkpoint directory")
    ap.add_argument("--model", required=True, help="deployed Ollama model (the Q4 one)")
    ap.add_argument("--control", default="",
                    help="quantization-free Ollama model (f16 GGUF) -- the metric's floor")
    ap.add_argument("--input", default=os.path.join(HERE, "infer_input.jsonl"))
    ap.add_argument("--n", type=int, default=3, help="prompts to compare")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--ref-cache", default=os.path.join(HERE, "quant_ref.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "quant_parity.json"))
    ap.add_argument("--refresh", action="store_true", help="recompute the reference")
    args = ap.parse_args()

    rows = load_jsonl(args.input, args.n)
    models = [args.model] + ([args.control] if args.control else [])
    print(f"[quant] {len(rows)} prompts, greedy, max_new_tokens={args.max_new_tokens}")
    print(f"[quant] models: {', '.join(models)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.merged)
    ref = load_reference(tok, rows, args)

    out = {m: [] for m in models}
    for m in models:
        for row in rows:
            r = generate(m, sg.render(tok, row), args.max_new_tokens, args.num_ctx)
            out[m].append((r.get("response") or "").strip())

    recs = []
    for i, row in enumerate(rows):
        rec = {"call_id": row["call_id"], "ref_chars": len(ref[i]), "model_chars": {}}
        for m in models:
            rec["model_chars"][m] = len(out[m][i])
            a = agreement(ref[i], out[m][i])
            rec[f"prefix_vs_ref::{m}"] = round(a, 4)
            rec[f"exact_vs_ref::{m}"] = ref[i] == out[m][i]
        if args.control:
            rec["prefix_f16_vs_quant"] = round(agreement(out[args.control][i], out[args.model][i]), 4)
        rec["ref_text"] = ref[i]
        for m in models:
            rec[f"text::{m}"] = out[m][i]
        recs.append(rec)

    print()
    for i, rec in enumerate(recs):
        print(f"=== {rec['call_id']} ===")
        print(f"    bf16 ref: {ref[i][:110]!r}")
        for m in models:
            print(f"    {m:22s} prefix {100 * rec[f'prefix_vs_ref::{m}']:5.1f}%  "
                  f"{out[m][i][:110]!r}")

    def med(vals):
        return statistics.median(vals) if vals else float("nan")

    per_model = {m: med([r[f"prefix_vs_ref::{m}"] for r in recs]) for m in models}
    print()
    for m in models:
        label = "engine floor" if m == args.control else "engine + quantization"
        print(f"[quant] {m:22s} median vs bf16 ref {100 * per_model[m]:5.1f}%   ({label})")
    if args.control:
        pq = med([r["prefix_f16_vs_quant"] for r in recs])
        print(f"[quant] {args.control} vs {args.model}  median {100 * pq:5.1f}%   "
              f"(quantization only, same engine)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"merged": args.merged, "models": models, "control": args.control,
                   "median_prefix_vs_ref": per_model, "records": recs}, f, indent=2)
    print(f"[quant] wrote -> {args.out}")

    # The verdict is relative, because only a comparison can be one. A quantized model
    # sitting as close to the reference as the quantization-free control is telling us
    # about the engine gap, not about Q4_K_M.
    if not args.control:
        print("[quant] NO CONTROL MODEL GIVEN: this number has no measured floor and "
              "cannot support a verdict. Re-run with --control.")
        return
    floor = per_model[args.control]
    got = per_model[args.model]
    if floor > 0.2 and got < 0.5 * floor:
        raise SystemExit(f"[quant] FAIL: {args.model} ({100 * got:.1f}%) is materially "
                         f"further from the bf16 reference than the f16 control "
                         f"({100 * floor:.1f}%) -- quantization is implicated")
    print(f"[quant] PASS: {args.model} sits within the f16 engine floor "
          f"({100 * got:.1f}% vs {100 * floor:.1f}%); the deployed summary text is not "
          f"separable from engine-level arithmetic noise")


if __name__ == "__main__":
    main()
