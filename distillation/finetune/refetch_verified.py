"""Re-fetch the shard in one pass, producing a verified file and a map of any damage.

The first download inherited a 1.877 GB partial from huggingface_hub and appended the
rest. That is a real hazard worth taking seriously -- a resume that appends at the wrong
offset corrupts silently and still leaves a file of the right length -- so this re-fetches
the whole thing under strict rails and compares it block by block against what was there.

It reported a sha256 mismatch, and corruption was the obvious reading. That reading was
wrong, and the evidence was this script's own bug. Two different hashes of one file sit
in the same API entry:

    lfs.oid  = aa33250c...   sha256 of the file's bytes (git-LFS semantics)
    xetHash  = 4a33a6bd...   Hugging Face's Xet content hash -- a different hash space

EXPECT_SHA had been transcribed from xetHash. Against the correct oid the shard verifies,
and all 543 blocks matched the previous file byte for byte: nothing was ever damaged. The
CDN's ETag is derived from the Xet path, which is why it appeared to disagree too.

What survives from that episode is the rail, which is worth keeping for any large resumed
fetch. The first downloader set a Range header and trusted whatever came back; it never
checked the status was 206 or that Content-Range started where it asked. A CDN answering
a ranged request with a 200, or from a different offset, would have had its body appended
in the wrong place. Every block here must be 206, must start at the offset requested, and
must arrive complete.

    python -I refetch_verified.py
"""
import argparse
import hashlib
import json
import os
import socket
import time
import urllib.error
import urllib.request

KD = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen3.5-2B"
SHARD = "model.safetensors-00001-of-00001.safetensors"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{SHARD}"
DEST = os.path.join(KD, "base_model", SHARD)
FRESH = DEST + ".fresh"
# The LFS oid -- the sha256 of the file's bytes. NOT the sibling `xetHash` field in the
# same API entry: the two are different hashes of the same file, and taking the wrong one
# makes a byte-perfect download look corrupt. GET /api/models/<repo>/tree/<rev> returns
# both; read `lfs.oid`.
EXPECT_SHA = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
BLOCK = 8 << 20
READ = 1 << 20
TRIES = 6


def fetch_block(start, end):
    """Exactly [start, end) of the published file, or raise."""
    want = end - start
    for attempt in range(1, TRIES + 1):
        try:
            req = urllib.request.Request(URL, headers={
                "User-Agent": "x", "Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                # The rails the first downloader was missing.
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}, expected 206")
                cr = r.headers.get("Content-Range") or ""
                if not cr.startswith(f"bytes {start}-"):
                    raise RuntimeError(f"Content-Range {cr!r} does not start at {start}")
                buf = bytearray()
                while len(buf) < want:
                    chunk = r.read(min(READ, want - len(buf)))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) != want:
                    raise RuntimeError(f"short body {len(buf)} != {want}")
                return bytes(buf)
        except Exception as e:  # noqa: BLE001 -- retry is the design
            if attempt == TRIES:
                raise
            print(f"[ref]   block {start:,} attempt {attempt}: "
                  f"{type(e).__name__}: {e}", flush=True)
            time.sleep(min(2 * attempt, 10))
    raise AssertionError("unreachable")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--fresh", default=FRESH)
    ap.add_argument("--no-compare", action="store_true",
                    help="skip comparing against the existing file")
    args = ap.parse_args()

    total = os.path.getsize(args.dest)
    print(f"[ref] {total / 1e9:.3f} GB, {total:,} bytes")
    print(f"[ref] pinning revision {REV[:12]}")
    print(f"[ref] writing {args.fresh}")

    old = None if args.no_compare else open(args.dest, "rb")
    differing = []
    t0 = time.time()
    done = 0

    with open(args.fresh, "wb") as out:
        start = 0
        while start < total:
            end = min(start + BLOCK, total)
            data = fetch_block(start, end)
            out.write(data)
            if old is not None:
                old.seek(start)
                prev = old.read(end - start)
                if prev != data:
                    # Narrow to the first differing byte for the record.
                    first = next((i for i in range(len(prev)) if prev[i] != data[i]),
                                 len(prev))
                    nz = sum(1 for i in range(first, len(prev)) if prev[i] != data[i])
                    differing.append({"start": start, "first_bad": start + first,
                                      "block_len": end - start, "diff_bytes": nz})
                    print(f"[ref] DIFFERS at {start:,} "
                          f"(first bad byte {start + first:,}, "
                          f"{nz:,} bytes of this block)", flush=True)
            done += end - start
            if done % (256 << 20) < BLOCK:
                rate = done / max(1e-9, time.time() - t0) / 1e6
                print(f"[ref] {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                      f"({100 * done / total:.0f}%) {rate:.1f} MB/s", flush=True)
            start = end

    if old is not None:
        old.close()

    size = os.path.getsize(args.fresh)
    if size != total:
        raise SystemExit(f"[ref] FATAL: {size} != {total}")
    print(f"[ref] complete in {(time.time() - t0) / 60:.1f} min; hashing...", flush=True)

    h = hashlib.sha256()
    with open(args.fresh, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    print(f"[ref] sha256 {got}")
    if got != EXPECT_SHA:
        raise SystemExit(f"[ref] FATAL: sha256 mismatch, expected {EXPECT_SHA}")
    print("[ref] sha256 verified against the published content hash")

    if differing:
        covered = sum(d["diff_bytes"] for d in differing)
        print(f"\n[ref] the previous file disagreed in {len(differing)} block(s), "
              f"{covered:,} bytes of {total:,} ({100 * covered / total:.4f}%)")
        json.dump(differing, open(os.path.join(KD, "shard_damage.json"), "w"), indent=2)
        print("[ref] wrote -> shard_damage.json")
    else:
        print("[ref] no differences: the previous file was byte-identical")


if __name__ == "__main__":
    main()
