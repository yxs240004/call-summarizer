"""Prove the merge changed nothing, at a resolution the merge can actually hold.

This file used to compare greedy token ids exactly, and that instrument cannot fail
for the right reason. The adapter computes Wx + B(Ax)*scale per token; the merged
checkpoint computes (W + BA*scale)x once. Those are equal in exact arithmetic and
different in bf16, so their argmaxes agree except where the reference's own top two
logits are within a hair. When they do disagree, one flip changes the context for
every later token, so every downstream comparison is answering a different question.
The old test reported "1/2 prompts differ" -- and the divergence was a tie at a margin
under 5e-5, while the merge's own effect on logits is up to ~0.2. It was measuring
arithmetic noise and calling it a merge defect.

So compare logits, teacher-forced on the same tokens, over every position of the
prompt, and read the result against the instrument's own resolution:

    adapter forward, twice, same input   -> the noise floor   (measured: 0.00000)
    adapter forward vs merged forward    -> the effect        (measured: KL ~5e-4)

A top-1 disagreement is then only acceptable where the effect's own size explains it:
a flip needs |d(top1) - d(top2)| > margin, and no per-logit delta can exceed max|dL|,
so a mismatch at a margin above 2*max|dL| means something other than rounding is
happening. That is a derived bound, not a hand-picked threshold.

    python -I verify_merged.py --adapter adapters/student-lora-blend --merged merged/student-blend
"""
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# -I implies -P, so the script's own directory is not on sys.path; add it to reuse the
# render function rather than re-writing it. Re-writing it is what caused the template
# defect this project already paid for once.
sys.path.insert(0, HERE)
import student_generate as sg  # noqa: E402

CHUNK = 64          # positions per slice: bounds peak RAM against a 248k-token vocab
KL_TOL = 1e-2       # 20x the measured 5e-4, so it catches breakage, not arithmetic


def load_jsonl(path, n):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n] if n else rows


def forward(model, tokenizer, row, max_length):
    """Next-token logits at EVERY position of the rendered prompt."""
    import torch
    enc = tokenizer(sg.render(tokenizer, row), return_tensors="pt",
                    truncation=True, max_length=max_length)
    with torch.no_grad():
        return model(**enc).logits[0]


def compare(a, b):
    """Per-position comparison of two [seq, vocab] logit blocks, chunked to bound RAM."""
    import torch
    seq = a.shape[0]
    maxabs, total, count, kls, rows = 0.0, 0.0, 0, [], []
    for s in range(0, seq, CHUNK):
        ax = a[s:s + CHUNK].float()
        bx = b[s:s + CHUNK].float()
        d = (ax - bx).abs()
        maxabs = max(maxabs, float(d.max()))
        total += float(d.sum())
        count += d.numel()

        sa = torch.softmax(ax, -1)
        sb = torch.softmax(bx, -1)
        kls.append(float((sa * (sa.clamp_min(1e-30).log()
                                - sb.clamp_min(1e-30).log())).sum(-1).mean()))

        ta = torch.topk(ax, 2, dim=-1)
        tb = torch.topk(bx, 2, dim=-1)
        for i in range(ax.shape[0]):
            rows.append((int(ta.indices[i, 0]), float(ta.values[i, 0]),
                         float(ta.values[i, 1]), int(tb.indices[i, 0])))
    return dict(maxabs=maxabs, meanabs=total / max(count, 1),
                kl=sum(kls) / max(len(kls), 1), rows=rows)


def from_adapter(base, adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    return tok, PeftModel.from_pretrained(model, adapter).eval()


def from_merged(path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16).eval()


def greedy(model, tokenizer, rows, max_new_tokens, max_length=4096):
    """Greedy-decode each row; return the generated ids per row.

    This lives here, beside the loaders, because verify_quant.py decodes the merged
    checkpoint exactly this way to build the bf16 reference it compares the deployed
    GGUF against -- and sharing one decode path is the point, for the reason in the
    header comment.

    main() below does NOT use it, and that is deliberate: comparing greedy argmaxes
    across the adapter's and the merged's arithmetic orders is the defect this file
    was rewritten to avoid. So this is a helper for its callers, not for the verdict.
    """
    import torch
    out = []
    for r in rows:
        enc = tokenizer(sg.render(tokenizer, r), return_tensors="pt",
                        truncation=True, max_length=max_length)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        out.append(ids[0, enc["input_ids"].shape[1]:].tolist())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=os.path.join(HERE, "base_model"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--input", default=os.path.join(HERE, "infer_input.jsonl"))
    ap.add_argument("--n", type=int, default=2, help="0 = all rows")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    rows = load_jsonl(args.input, args.n)
    print(f"[verify] {len(rows)} prompts, teacher-forced logits, "
          f"max_length={args.max_length}", flush=True)

    # The merged checkpoint must render the same prompt the training run used. If the
    # template did not travel with the merge, everything below would compare two
    # different questions.
    from transformers import AutoTokenizer
    mtok = AutoTokenizer.from_pretrained(args.merged)
    tail = sg.marker_tail(mtok)
    got = mtok(sg.render(mtok, rows[0]), add_special_tokens=False)["input_ids"][-len(tail):]
    if got != tail:
        raise SystemExit("[verify] FATAL: merged checkpoint lost the template")
    print("[verify] merged prompt ends with the training block: True", flush=True)

    t0 = time.time()
    tok_a, model_a = from_adapter(args.base, args.adapter)
    tok_m, model_m = from_merged(args.merged)
    print(f"[verify] both checkpoints loaded in {time.time() - t0:.0f}s", flush=True)

    bad = 0
    for i, row in enumerate(rows):
        t0 = time.time()
        a1 = forward(model_a, tok_a, row, args.max_length)
        floor = compare(a1, forward(model_a, tok_a, row, args.max_length))
        gc.collect()
        effect = compare(a1, forward(model_m, tok_m, row, args.max_length))
        del a1
        gc.collect()

        print()
        print(f"=== {row['call_id']}  ({effect['rows'].__len__()} positions, "
              f"{time.time() - t0:.0f}s) ===")
        print(f"  noise floor (adapter vs adapter)  max|dL| {floor['maxabs']:.5f}  "
              f"mean|dL| {floor['meanabs']:.6f}  KL {floor['kl']:.3e}")
        mism = [(p, r) for p, r in enumerate(effect["rows"]) if r[0] != r[3]]
        n = len(effect["rows"])
        print(f"  effect      (adapter vs merged )  max|dL| {effect['maxabs']:.5f}  "
              f"mean|dL| {effect['meanabs']:.6f}  KL {effect['kl']:.3e}")
        print(f"  top-1 agreement: {n - len(mism)}/{n} "
              f"({100 * (n - len(mism)) / n:.2f}%)")

        if floor["maxabs"] != 0.0:
            print(f"  NOTE: repeated forwards are not bit-identical "
                  f"(floor {floor['maxabs']:.3e}); that is the resolution limit.")
        bound = 2 * max(effect["maxabs"], floor["maxabs"])
        unexplained = [(p, r) for p, r in mism if (r[1] - r[2]) > bound]
        for p, r in mism:
            print(f"      mismatch @pos {p}: reference margin {r[1] - r[2]:.5f} "
                  f"(logits {r[1]:.4f} vs {r[2]:.4f})")
        if unexplained:
            bad += 1
            print(f"  FAIL: {len(unexplained)} mismatch(es) above the rounding bound "
                  f"{bound:.5f} -- merge arithmetic does not explain them")
        elif mism:
            print(f"  ok: all {len(mism)} mismatch(es) inside the rounding bound "
                  f"{bound:.5f}")
        else:
            print(f"  ok: top-1 identical at all {n} positions")

        if effect["kl"] > KL_TOL:
            bad += 1
            print(f"  FAIL: KL {effect['kl']:.3e} exceeds {KL_TOL:.0e}")

    print()
    if bad:
        raise SystemExit(f"[verify] FAIL: {bad} prompt(s) differ by more than merge "
                         f"arithmetic explains")
    print("[verify] PASS: the merged checkpoint is the adapter's function, to within "
          "bf16 merge arithmetic")


if __name__ == "__main__":
    main()
