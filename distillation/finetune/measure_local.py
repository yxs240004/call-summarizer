"""Measure the distilled student on this laptop, splitting prefill from decode.

D9's open risk is that prompt processing dominates CPU latency, and there was no x86
laptop measurement at all -- the quoted figures were an RK3588's. Ollama reports
prompt_eval_duration and eval_duration separately, which is exactly that split.

Prompts are rendered through the student's own tokenizer and sent with raw=true, so the
numbers describe the model rather than Ollama's idea of the template, and are directly
comparable to the HF runs. --check-template then exercises the templated path and compares
Ollama's prompt_eval_count against the tokenizer's own count, which is how the deployed
template is verified rather than trusted.

    python -I measure_local.py --model student-blend --out local_blend.jsonl
    python -I measure_local.py --model student-blend --check-template

WHAT THIS FILE USED TO GET WRONG. It asked Ollama to keep the model resident for 30
minutes and never accounted for what that costs. Ollama reuses the KV cache of a prompt
prefix across requests -- including across processes -- so a prompt an earlier run had
already sent comes back with prompt_eval_duration near zero: 0.07-0.12 s for 626-1,764
tokens, an implied 10,000+ tok/s, which this CPU cannot do (llama-bench measures 206
tok/s prefill on the same file). The tool then reported "median call 4.0s, 3% of it
prefill" over three prompts that had all been cached by a previous run, when the true
fresh cost is 12-16 s at 55-60% prefill. The default is now keep_alive=0, so every call
pays its own prefill and the split means what it says. --keep-alive 30m restores the
cached behaviour, which is the right way to measure *re*-summarizing a call but not a
call. A guard also flags any prefill implying more than 2,000 tok/s, because that is a
cache hit rather than a measurement.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import student_generate as sg  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def post(path, payload, timeout=1800):
    req = urllib.request.Request(OLLAMA + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def options(num_predict, num_ctx, greedy=False):
    if greedy:
        # Argmax, no repetition penalty, no nucleus filter. NOT a match for the HF arm:
        # that samples at 0.7/0.9/1.05 like the shipped run, so greedy is a new
        # condition rather than a reconstruction of it. Run it to ask whether the
        # ungrounded names are a sampling artifact and whether the pipeline can be made
        # deterministic; not to compare precision against the HF arm.
        return {
            "temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.0,
            "num_predict": num_predict, "num_ctx": num_ctx, "seed": 42,
        }
    return {
        "temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.05,
        "num_predict": num_predict, "num_ctx": num_ctx, "seed": 42,
    }


def check_template(model, tok, row, num_ctx):
    """Does the deployed template produce the prompt we think it does?

    Ollama does not return the rendered prompt, so this compares the reported
    prompt_eval_count against the tokenizer's own count for the same render. Equal counts
    are strong evidence the template travelled into the GGUF and rendered identically; a
    difference means the deployed prompt is not the one that was trained on.
    """
    rendered = sg.render(tok, row)
    want = len(tok(rendered, add_special_tokens=False)["input_ids"])
    resp = post("/api/chat", {
        "model": model, "stream": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": row["system"]},
                     {"role": "user", "content": row["user"]}],
        "options": options(16, num_ctx),
    })
    got = resp.get("prompt_eval_count")
    print(f"[template] hf={want} ollama={got} match={want == got}")
    print(f"[template] ollama replied: {resp.get('message', {}).get('content', '')[:160]!r}")
    return want == got


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--input", default=os.path.join(HERE, "infer_input.jsonl"))
    ap.add_argument("--adapter", default=os.path.join(HERE, "adapters", "student-lora-blend"),
                    help="tokenizer source; the chat template must be the trained one")
    ap.add_argument("--out", default=os.path.join(HERE, "local_samples.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=900)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--check-template", action="store_true")
    ap.add_argument("--keep-alive", default="0",
                    help="Ollama keep_alive; 0 unloads after each call so no prompt "
                         "prefix is cached from a previous run")
    ap.add_argument("--greedy", action="store_true",
                    help="decode by argmax (temperature 0, repeat penalty 1.0, top_p "
                         "1.0) instead of the deployed 0.7/0.9/1.05. This is NOT what the "
                         "HF/eval arms do -- they sample at 0.7 -- so a greedy run asks a "
                         "new question: is the invention a sampling artifact, and can the "
                         "pipeline be made deterministic?")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]

    if args.check_template:
        ok = check_template(args.model, tok, rows[0], args.num_ctx)
        raise SystemExit(0 if ok else 1)

    print(f"[local] {len(rows)} calls through Ollama model {args.model!r} "
          f"(keep_alive={args.keep_alive})")
    ka = int(args.keep_alive) if args.keep_alive.isdigit() else args.keep_alive
    out = open(args.out, "w", encoding="utf-8")
    t_start = time.time()
    pre, dec = [], []
    hits = 0
    flags = []
    for i, row in enumerate(rows, 1):
        prompt = sg.render(tok, row)
        t0 = time.time()
        r = post("/api/generate", {
            "model": args.model, "prompt": prompt, "raw": True, "stream": False,
            "keep_alive": ka,
            "options": options(args.num_predict, args.num_ctx, greedy=args.greedy),
        })
        wall = time.time() - t0
        # Ollama reports nanoseconds.
        p = r.get("prompt_eval_duration", 0) / 1e9
        d = r.get("eval_duration", 0) / 1e9
        n_pt = r.get("prompt_eval_count") or 0
        # A prefill this fast is a cache hit, not a measurement: see the docstring. It
        # means this prompt's prefix was already resident, so the 12-16 s figure does not
        # apply to it and a median over such rows understates the real cost.
        from_cache = bool(p > 0 and n_pt / p > 2000)
        hits += from_cache
        flags.append(from_cache)
        pre.append(p)
        dec.append(d)
        rec = {
            "call_id": row["call_id"],
            "summary": (r.get("response") or "").strip(),
            "model": args.model,
            "decoding": "greedy" if args.greedy else "temp0.7_top_p0.9_rp1.05",
            "prompt_tokens": r.get("prompt_eval_count"),
            "prompt_tokens_cached": r.get("prompt_eval_cached_count"),
            "output_tokens": r.get("eval_count"),
            "prefill_s": round(p, 2),
            "decode_s": round(d, 2),
            "load_s": round(r.get("load_duration", 0) / 1e9, 2),
            "wall_s": round(wall, 2),
            "done_reason": r.get("done_reason"),
            "from_cache": from_cache,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if i <= 5 or i % 10 == 0:
            share = 100 * p / (p + d) if (p + d) else 0
            print(f"[{i}/{len(rows)}] {row['call_id']} prefill {p:.1f}s decode {d:.1f}s "
                  f"({share:.0f}% prefill) {rec['output_tokens']} tok "
                  f"{(r.get('eval_count', 0) / d if d else 0):.1f} tok/s")
    out.close()

    total = time.time() - t_start
    print(f"\n[local] {len(rows)} calls in {total / 60:.1f} min ({total / len(rows):.1f}s/call)\n")
    print(f"{'':<10}{'median':>9}{'p90':>9}{'mean':>9}{'max':>9}")
    for name, vals in (("prefill", pre), ("decode", dec)):
        s = sorted(vals)
        p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
        print(f"{name:<10}{statistics.median(vals):>9.1f}{p90:>9.1f}"
              f"{statistics.fmean(vals):>9.1f}{max(vals):>9.1f}")
    mp, md = statistics.median(pre), statistics.median(dec)
    print(f"\n[local] median call: {mp + md:.1f}s, {100 * mp / (mp + md):.0f}% of it prefill")
    if hits:
        f_pre = [p for p, c in zip(pre, flags) if not c]
        f_dec = [d for d, c in zip(dec, flags) if not c]
        if f_pre:
            fm_p, fm_d = statistics.median(f_pre), statistics.median(f_dec)
            print(f"[local] WARNING: {hits}/{len(rows)} calls were served from a cached prompt "
                  f"prefix (impossible prefill); they are excluded from the fresh median below")
            print(f"[local] fresh-call median over {len(f_pre)}: {fm_p + fm_d:.1f}s, "
                  f"{100 * fm_p / (fm_p + fm_d):.0f}% of it prefill")
    print(f"[local] wrote -> {args.out}")


if __name__ == "__main__":
    main()
