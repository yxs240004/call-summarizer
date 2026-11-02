"""Map exactly which byte ranges of the shard disagree with the remote.

The zero scan came back clean, so this is not a sparse hole: the bytes are present and
wrong. That leaves a splice -- some region holding content from somewhere other than
where it belongs, at the right length. The cheapest way to see its shape is to probe
evenly spaced windows and compare, then bisect the boundaries of any bad run.

Every probe re-requests the resolve URL, which redirects to the content-addressed CDN
object named by the same sha256 the local file fails to hash to, so "differ" here really
does mean the local bytes are not the published ones.

    python -I pinpoint_corruption.py --points 21
    python -I pinpoint_corruption.py --refine
"""
import argparse
import os
import urllib.request

KD = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen3.5-2B"
SHARD = "model.safetensors-00001-of-00001.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
DEST = os.path.join(KD, "base_model", SHARD)
WIN = 64 << 10

_cache = {}


def ranged(offset, length):
    req = urllib.request.Request(URL, headers={
        "User-Agent": "x", "Range": f"bytes={offset}-{offset + length - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def probe(off, total):
    """Does the local file agree with the remote at this offset?"""
    if off in _cache:
        return _cache[off]
    win = min(WIN, total - off)
    if win <= 0:
        return True
    with open(DEST, "rb") as f:
        f.seek(off)
        local = f.read(win)
    remote = ranged(off, win)
    if len(remote) != len(local):
        raise RuntimeError(f"short read at {off}: {len(remote)} != {len(local)}")
    _cache[off] = local == remote
    return _cache[off]


def grid(total, points):
    return sorted({int(i * (total - WIN) / max(1, points - 1)) for i in range(points)})


def scan_grid(total, points):
    bad = []
    for off in grid(total, points):
        ok = probe(off, total)
        if not ok:
            bad.append(off)
        print(f"[pin] {'ok  ' if ok else 'BAD '} {off:>13,} ({100 * off / total:5.1f}%)",
              flush=True)
    return bad


def bisect_edge(total, good, bad):
    """Boundary between a matching and a non-matching offset, to WIN precision."""
    while abs(bad - good) > WIN:
        mid = good + ((bad - good) // 2 // WIN) * WIN
        if mid in (good, bad):
            break
        if probe(mid, total):
            good = mid
        else:
            bad = mid
    return good, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=int, default=21)
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()

    total = os.path.getsize(DEST)
    print(f"[pin] local {DEST}\n[pin] {total:,} bytes\n")

    bad = scan_grid(total, args.points)
    print(f"\n[pin] {len(bad)} of {args.points} probes disagree")
    if not bad:
        print("[pin] every sampled offset matches -- corruption is smaller than the "
              "probe spacing, or the hash reference is wrong")
        return

    # The interesting shape is whether the bad probes form one contiguous run to EOF.
    badset = set(bad)
    last = max(bad)
    if last >= total - WIN - (total // args.points):
        print("[pin] the final probe disagrees: the run extends to the end of the file")
    print(f"[pin] first bad probe {min(bad):,}  last bad probe {last:,}")
    print(f"[pin] bad fraction of probes: {100 * len(badset) / args.points:.0f}%")

    if args.refine:
        print("\n[pin] refining the leading edge")
        good_pt = max([o for o in grid(total, args.points) if probe(o, total) and o < min(bad)],
                      default=0)
        g, b = bisect_edge(total, good_pt, min(bad))
        print(f"[pin] local correct up to {g:,}, first bad byte in "
              f"({g:,}, {b:,}]  ->  {b:,} ({b / 1e9:.4f} GB)")


if __name__ == "__main__":
    main()
