"""Prove the deployed model actually RUNS -- not merely that it was created.

This stage exists because of a specific failure that this chain reported as success.
The converter had declared `block_count = 25` (the merged config says
`mtp_num_hidden_layers = 1`) while the merge had dropped the 15 `mtp.*` tensors, so
block 24 was declared and never written. `llama-quantize` quantized it happily and
`ollama create` printed `success`, and the whole chain exited 0 -- and then the first
real request returned

    HTTP 500 check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found

Neither of those stages builds a compute graph, so neither can see a missing tensor.
Only asking for tokens can. A model that does not load is indistinguishable from a
good one from the GGUF side, which is why this is a stage and not a manual check.

Deliberately cheap: a short prompt and 16 tokens, so it costs seconds, not minutes.

    python -I smoke_deployed.py --model student-blend
"""
import argparse
import json
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="student-blend")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="seconds to wait for the first token (cold model load)")
    ap.add_argument("--num-predict", type=int, default=16)
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "prompt": "The meeting summary is",
        "raw": True,          # no template: this asks whether the weights run at all
        "stream": False,
        "options": {"num_predict": args.num_predict, "num_ctx": 512, "temperature": 0},
    }
    req = urllib.request.Request(
        args.host + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=args.timeout).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:1000]
        raise SystemExit(f"[smoke] FAIL: the deployed model does not run "
                         f"(HTTP {e.code}):\n{body}")

    text = d.get("response", "")
    generated = int(d.get("eval_count", 0) or 0)
    print(f"[smoke] {args.model}: prompt {d.get('prompt_eval_count')} tokens, "
          f"generated {generated} tokens, done_reason={d.get('done_reason')}")
    print(f"[smoke] text: {text[:200]!r}")

    # The point of the stage: zero tokens is a failure even though every earlier stage
    # called this a success.
    if generated == 0 or not text.strip():
        raise SystemExit("[smoke] FAIL: the server returned no tokens -- the model "
                         "loaded but does not generate")
    print("[smoke] PASS: the deployed model loads and generates tokens")


if __name__ == "__main__":
    main()
