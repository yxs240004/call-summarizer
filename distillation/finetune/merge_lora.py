"""Fold a trained student LoRA into Qwen3.5-2B, producing a standalone checkpoint.

The GGUF converter reads a plain HF directory, so the adapter has to be merged in
first. Merging is arithmetic on bf16 weights, so it is not assumed harmless --
verify_merged.py compares the two checkpoints' next-token distributions.

Two properties of the base are easy to destroy here, both silently:

  * dtype policy. The base stores 36 tensors as F32 (18 x linear_attn.A_log,
    18 x linear_attn.norm.weight) and its config declares
    mamba_ssm_dtype="float32". Loading with dtype=torch.bfloat16 rounds them, and
    save_pretrained then writes the whole model in one dtype, so the declaration
    stops being true of the file. Measured cost of that rounding: 0.000e+00 on
    A_log (this checkpoint's A_log values are exactly bf16-representable) and up
    to 3.9e-3 relative on norm.weight. Restored below, exactly, from the base.

  * the MTP head. AutoModelForImageTextToText does not own the mtp.* tensors, so
    they are dropped at load and never written. Plain autoregressive decoding
    does not use them, but the checkpoint's structure changes, so it is reported
    rather than left to be inferred from a file size.

Neither A_log nor linear_attn.norm.weight is a LoRA target, so the adapter never
touched them and their correct merged value IS the base value -- which is why the
restore is exact rather than approximate, and why the rail below can be an
equality assertion instead of a tolerance.

    python -I merge_lora.py --adapter adapters/student-lora-blend --out merged/student-blend
"""
import argparse
import glob
import json
import os
import shutil
import struct
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def base_header(base_dir):
    """The base shard's tensor names and dtypes, read from the safetensors header."""
    shards = glob.glob(os.path.join(base_dir, "*.safetensors"))
    if len(shards) != 1:
        raise SystemExit(f"[merge] expected one safetensors shard in {base_dir}, "
                         f"found {len(shards)}")
    with open(shards[0], "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return shards[0], header


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=os.path.join(HERE, "base_model"),
                    help="the un-adapted Qwen/Qwen3.5-2B directory")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    base_path, base_hdr = base_header(args.base)
    f32_names = sorted(k for k, v in base_hdr.items() if v["dtype"] == "F32")
    print(f"[merge] base declares {len(f32_names)} F32 tensors "
          f"({sum('A_log' in k for k in f32_names)} A_log, "
          f"{sum('norm.weight' in k for k in f32_names)} norm.weight)")

    # The tokenizer comes from the adapter, not the base: the adapter directory is the
    # one that carries the student's own chat template, and the template is the half of
    # this job that has already produced one silent corruption.
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    print(f"[merge] tokenizer <- {args.adapter} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    # dtype, not torch_dtype: transformers 5.x silently ignores the old spelling and
    # hands back fp32, which doubles the file and is not what trained.
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    print(f"[merge] base <- {args.base} ({time.time() - t0:.1f}s)")

    model = PeftModel.from_pretrained(model, args.adapter)
    attached = [n for n, _ in model.named_parameters() if "lora_" in n]
    print(f"[merge] adapter attached: {len(attached)} lora tensors")

    merged = model.merge_and_unload()

    # merge_and_unload is supposed to return a plain model. If any adapter tensor
    # survived, the checkpoint would carry a half-merged model and every downstream
    # number would be quietly wrong -- so this is checked, not trusted.
    leftover = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if leftover:
        raise SystemExit(f"[merge] FATAL: {len(leftover)} lora tensors survived "
                         f"merge_and_unload, e.g. {leftover[:3]}")

    # Restore the base's F32 tensors. Nothing above is allowed to have changed them --
    # they are not LoRA targets -- so the base's exact value is the right one, and the
    # rail is an equality check rather than a tolerance. If a future adapter ever does
    # target one of these, this raises instead of quietly discarding adapter output.
    restored = []
    with safe_open(base_path, framework="pt") as bf:
        for name, tensor in list(merged.named_parameters()) + list(merged.named_buffers()):
            if name not in f32_names or name not in bf.keys():
                continue
            exact = bf.get_tensor(name).to(torch.float32)
            if not torch.equal(tensor.data.float(), exact.to(torch.bfloat16).float()):
                raise SystemExit(
                    f"[merge] FATAL: {name} is F32 in the base but the merge changed it. "
                    f"Restoring the base value would discard adapter output; refusing.")
            tensor.data = exact
            restored.append(name)
    if len(restored) != len(f32_names):
        missing = sorted(set(f32_names) - set(restored))
        raise SystemExit(f"[merge] FATAL: {len(missing)} F32 tensors not restored, "
                         f"e.g. {missing[:3]}")
    print(f"[merge] restored {len(restored)} tensors to the base's F32 "
          f"(dtype policy preserved)")

    # Report what the model class does not own, instead of letting it show up only as
    # an unexplained gap in the file size.
    written = set(dict(merged.named_parameters())) | set(dict(merged.named_buffers()))
    dropped = sorted(set(base_hdr) - written)
    if dropped:
        print(f"[merge] NOTE: {len(dropped)} base tensors are not owned by "
              f"{type(merged).__name__} and will not be written, e.g. {dropped[:4]}")
        if all(n.startswith("mtp.") for n in dropped):
            print("[merge]       all of them are mtp.* (multi-token prediction); "
                  "plain autoregressive decoding does not use these.")

    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    src = os.path.join(args.adapter, "chat_template.jinja")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(args.out, "chat_template.jinja"))
        print("[merge] carried chat_template.jinja across")

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out) if f.endswith(".safetensors"))
    print(f"[merge] wrote {args.out} ({total / 1e9:.2f} GB of safetensors)")

    # Read the artifact back rather than trusting what was just written: this is the
    # property that was silently wrong before, so it is verified, not printed.
    _, out_hdr = base_header(args.out)
    out_f32 = sorted(k for k, v in out_hdr.items() if v["dtype"] == "F32")
    print(f"[merge] artifact carries {len(out_hdr)} tensors, {len(out_f32)} of them F32")
    if out_f32 != f32_names:
        raise SystemExit(f"[merge] FATAL: artifact F32 set differs from the base's "
                         f"({len(out_f32)} vs {len(f32_names)}); the dtype policy did "
                         f"not survive the save")


if __name__ == "__main__":
    main()
