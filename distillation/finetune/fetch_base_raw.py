"""Download the Qwen3.5-2B weight shard with ranged HTTP requests, resuming properly.

huggingface_hub kept abandoning its own partial file and opening a fresh zero-byte one on
every retry, and hf_transfer is deprecated in favour of Xet, so both clients were noise
here. The shard is a single 4.55 GB file, which makes a ranged downloader the whole job:
resume from a verified offset, keep a real socket timeout so a wedged connection raises
instead of hanging, and check the result against the Hub's advertised sha256 at the end.

The offset is only trusted after the local prefix is compared byte-for-byte against the
remote content, because a partial file from a different revision would otherwise be
glued to the wrong bytes and produce a model that loads and is silently wrong.

    python -I fetch_base_raw.py
"""
import hashlib
import os
import shutil
import socket
import time
import urllib.error
import urllib.request

KD = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen3.5-2B"
SHARD = "model.safetensors-00001-of-00001.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
DEST = os.path.join(KD, "base_model", SHARD)
CACHE = os.path.join(KD, "base_model", ".cache", "huggingface", "download")
CHUNK = 4 << 20
TIMEOUT = 30
VERIFY_PREFIX = 1 << 20


def remote_head():
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "x"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        etag = (r.headers.get("ETag") or "").strip('"')
        return etag, int(r.headers["Content-Length"])


def ranged(offset, length=None):
    headers = {"User-Agent": "x", "Range": f"bytes={offset}-"}
    if length is not None:
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
    return urllib.request.urlopen(
        urllib.request.Request(URL, headers=headers), timeout=TIMEOUT)


def prefix_is_valid(path, offset):
    """Is the local prefix really the head of the remote file?"""
    n = min(VERIFY_PREFIX, offset)
    if n == 0:
        return True
    with open(path, "rb") as f:
        local = f.read(n)
    with ranged(0, n) as r:
        remote = r.read()
    if local != remote:
        print(f"[raw] prefix mismatch over {n} bytes -- discarding the partial file")
        return False
    print(f"[raw] prefix verified: first {n} bytes match the remote shard")
    return True


def main():
    etag, total = remote_head()
    print(f"[raw] {SHARD}: {total / 1e9:.2f} GB, sha256 {etag[:16]}...")

    partials = [os.path.join(CACHE, f) for f in os.listdir(CACHE)
                if f.endswith(".incomplete")] if os.path.isdir(CACHE) else []
    partials = [p for p in partials if os.path.getsize(p) > 0]
    if partials:
        src = max(partials, key=os.path.getsize)
        size = os.path.getsize(src)
        print(f"[raw] found existing partial: {size / 1e9:.2f} GB")
        if prefix_is_valid(src, size):
            if not os.path.exists(DEST):
                shutil.move(src, DEST)
            offset = os.path.getsize(DEST)
        else:
            offset = 0
    else:
        offset = os.path.getsize(DEST) if os.path.exists(DEST) else 0

    if offset >= total:
        print("[raw] already complete")
    else:
        print(f"[raw] resuming at {offset / 1e9:.2f} GB "
              f"({(total - offset) / 1e6:.0f} MB to go)")

    failures = 0
    t0 = time.time()
    last_report = offset
    while offset < total:
        try:
            with ranged(offset) as r, open(DEST, "ab") as f:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    offset += len(block)
                    if offset - last_report >= 200 << 20:
                        print(f"[raw] {offset / 1e9:.2f}/{total / 1e9:.2f} GB "
                              f"({100 * offset / total:.0f}%)", flush=True)
                        last_report = offset
            failures = 0
        except (socket.timeout, urllib.error.URLError, ConnectionError, OSError) as e:
            failures += 1
            print(f"[raw] {type(e).__name__} at {offset / 1e9:.2f} GB "
                  f"(failure {failures})", flush=True)
            if failures > 20:
                raise SystemExit("[raw] too many failures")
            time.sleep(min(2 * failures, 15))

    size = os.path.getsize(DEST)
    if size != total:
        raise SystemExit(f"[raw] FATAL: {size} bytes on disk, expected {total}")

    print(f"[raw] downloaded {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min; "
          f"hashing...", flush=True)
    h = hashlib.sha256()
    with open(DEST, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if got != etag:
        raise SystemExit(f"[raw] FATAL: sha256 {got[:16]}... != {etag[:16]}...")
    print(f"[raw] sha256 verified: {got[:16]}...")

    shutil.rmtree(CACHE, ignore_errors=True)
    print(f"[raw] DONE {DEST}")


if __name__ == "__main__":
    main()
