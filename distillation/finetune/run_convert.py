"""Run llama.cpp's HF->GGUF converter from an isolated interpreter.

`-I` keeps the current directory and the script's own directory off sys.path, which is
what makes it safe to point an interpreter at a downloaded tree -- but it also means the
converter's `from conversion import ...` cannot resolve, because that expects its own
directory on sys.path. So the two paths it genuinely needs are added explicitly and
nothing else is. `-I` implies -E, so PYTHONPATH is ignored and this wrapper is the only
way to reach the repository's own gguf-py from an isolated interpreter.
"""
import argparse
import os
import runpy
import sys

KD = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(KD, "llama.cpp")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="merged HF checkpoint directory")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--outtype", default="f16")
    ap.add_argument("--no-mtp", action="store_true",
                    help="tell the converter the checkpoint carries no NextN/MTP head")
    args = ap.parse_args()

    # The repository's gguf-py must win over any pip-installed gguf. The converter and
    # the gguf-py it ships with are one commit, and only that pairing is guaranteed to
    # know about newer architectures such as QWEN35.
    sys.path.insert(0, os.path.join(LLAMA, "gguf-py"))
    sys.path.insert(0, LLAMA)

    sys.argv = ["convert_hf_to_gguf.py", args.model,
                "--outfile", args.outfile, "--outtype", args.outtype]
    # Why this flag has to exist, in one paragraph, because the failure it prevents is
    # invisible from the GGUF side. The converter computes
    #     block_count = num_hidden_layers + mtp_num_hidden_layers
    # whenever `no_mtp` is False, and it only sets no_mtp itself when
    # `text_config.architectures` names Qwen3_5TextModel -- which this checkpoint's
    # config does not (it is None in the base AND in the merge). So the converter
    # declares 25 blocks. The base ships its 15 mtp.* tensors and is fine; the merge
    # does not (AutoModelForImageTextToText does not own them), so block 24 is declared
    # and never written, and llama.cpp dies at graph construction with
    #     check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found
    # -- while llama-quantize and `ollama create` both report success, because neither
    # builds a compute graph. Hence the flag, and hence the generation stage after it.
    if args.no_mtp:
        sys.argv.append("--no-mtp")
    runpy.run_path(os.path.join(LLAMA, "convert_hf_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
