"""Why did the shard's sha256 not match? Cheap questions first.

The download produced a file of exactly the right length and a hash matching nothing.
Two shapes produce that: the remote `main` revision moved during a 222-minute transfer
so the file splices two revisions, or a resume wrote bytes from the wrong offset.

The Hub's model API is the authority here -- each LFS sibling carries the sha256 of its
content as `oid`, which is what ETag on the CDN sometimes is and sometimes is not. So
this asks the API for the current revision and the file's real oid, and compares all
three without fetching any weights.

    python -I diagnose_shard.py
"""
import json
import os
import urllib.error
import urllib.request

KD = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen3.5-2B"
SHARD = "model.safetensors-00001-of-00001.safetensors"
DEST = os.path.join(KD, "base_model", SHARD)
RECORDED_ETAG = "4a33a6bddec714d884625c359882b7169abdd117115a8da6322a2f117c55826a"
RECORDED_SIZE = 4548221488


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "x"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def head(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "x"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return (r.headers.get("ETag") or "").strip('"'), \
               int(r.headers.get("Content-Length", 0)), r.geturl()


def main():
    info = get_json(f"https://huggingface.co/api/models/{REPO}")
    rev = info["sha"]
    print(f"[diag] current main revision: {rev}")

    sib = next((s for s in info["siblings"] if s["rfilename"] == SHARD), None)
    if sib is None:
        raise SystemExit(f"[diag] {SHARD} is not in the repo listing")
    lfs = sib.get("lfs") or {}
    oid = lfs.get("oid") or lfs.get("sha256")
    print(f"[diag] repo listing size   : {lfs.get('size')}")
    print(f"[diag] repo listing sha256 : {oid}")

    for label, url in (("main", f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"),
                       ("pinned", f"https://huggingface.co/{REPO}/resolve/{rev}/{SHARD}")):
        try:
            etag, size, final = head(url)
        except urllib.error.HTTPError as e:
            print(f"[diag] HEAD {label}: HTTP {e.code}")
            continue
        print(f"[diag] HEAD {label}: content-length {size}, etag {etag}")
        print(f"[diag]   landed on {final}")

    actual = os.path.getsize(DEST) if os.path.exists(DEST) else -1
    print()
    print(f"[diag] on disk: {actual} bytes (recorded {RECORDED_SIZE}, "
          f"{'same' if actual == RECORDED_SIZE else 'DIFFERENT'})")
    print(f"[diag] recorded etag: {RECORDED_ETAG}")
    if oid:
        print(f"[diag] VERDICT: recorded etag "
              f"{'MATCHES' if oid == RECORDED_ETAG else 'DIFFERS FROM'} the repo's "
              f"current sha256 for this file")
        print(f"[diag]   -> the remote revision "
              f"{'did NOT move' if oid == RECORDED_ETAG else 'MOVED (or the CDN etag is not the content hash)'}"
              f" during the transfer")


if __name__ == "__main__":
    main()
