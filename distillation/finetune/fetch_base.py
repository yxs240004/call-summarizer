"""Fetch the un-adapted Qwen3.5-2B weights, resuming across stalls.

snapshot_download resumes from the .incomplete file, so a retry costs only the bytes it
re-fetches -- no progress is lost. The first attempt wedged at 1.88 GB of 4.55 GB with the
process still alive and the socket dead, and huggingface_hub surfaces no exception for
that, so the retry loop is the whole point of this script. HF_HUB_DOWNLOAD_TIMEOUT makes a
dead socket raise instead of hanging forever.

    python -I fetch_base.py
"""
import os
import time

# Set here rather than in the shell: this script runs under -I, which implies -E, so
# exported environment variables would be ignored.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
try:
    # Parallel chunked transfer. The single-connection rate measured 0.57 MB/s against a
    # 4.55 GB shard, which is over two hours; this is the lever that moves.
    import hf_transfer  # noqa: F401
except ImportError:
    print("[fetch] hf_transfer not installed -- falling back to the serial downloader",
          flush=True)
else:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("[fetch] hf_transfer enabled", flush=True)

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "Qwen/Qwen3.5-2B"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_model")
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def partial_bytes():
    """Bytes already pulled down, from the .incomplete file and finished shards."""
    total = 0
    cache = os.path.join(DEST, ".cache", "huggingface", "download")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            if f.endswith(".incomplete"):
                total += os.path.getsize(os.path.join(cache, f))
    for f in os.listdir(DEST) if os.path.isdir(DEST) else []:
        if f.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(DEST, f))
    return total


def main():
    for attempt in range(1, 9):
        before = partial_bytes()
        print(f"[fetch] attempt {attempt}: {before / 1e9:.2f} GB on disk", flush=True)
        try:
            path = snapshot_download(REPO, local_dir=DEST, allow_patterns=PATTERNS)
        except Exception as e:  # noqa: BLE001 -- the retry is the design
            print(f"[fetch] attempt {attempt} raised {type(e).__name__}: {e}", flush=True)
        else:
            print(f"[fetch] DONE {path}", flush=True)
            return
        after = partial_bytes()
        print(f"[fetch] attempt {attempt} ended at {after / 1e9:.2f} GB "
              f"(+{(after - before) / 1e6:.0f} MB)", flush=True)
        time.sleep(3)
    raise SystemExit("[fetch] gave up after 8 attempts")


if __name__ == "__main__":
    main()
