"""Find where the shard is wrong, cheaply, before re-downloading anything.

Same length, wrong sha256. The suspicion is that the inherited 1.877 GB partial from
huggingface_hub contains sparse holes: that client writes chunks at random offsets into
a pre-sized file, so a wedged transfer leaves regions that were never written -- which
read back as zeros while the file length stays exactly right. A 1 MB prefix check would
walk straight over one, which is why the prefix verified and the hash did not.

Zero regions are found locally, with no network at all, because real bf16 weights do not
contain megabyte runs of 0x00. Those regions are then confirmed against the remote bytes
at the same offsets, and a coarse remote sweep catches any corruption that is not zeros.

    python -I locate_corruption.py            # local zero scan only
    python -I locate_corruption.py --remote    # + confirm against the CDN
"""
import argparse
import os
import urllib.request

KD = os.path.dirname(os.path.abspath(__file__))
REPO = "Qwen/Qwen3.5-2B"
SHARD = "model.safetensors-00001-of-00001.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
DEST = os.path.join(KD, "base_model", SHARD)

BLOCK = 1 << 20          # 1 MB scan block
ZERO_FRAC = 0.98         # a block this empty is a hole, not a weight
SWEEP = 32 << 20         # coarse remote sweep step
WIN = 64 << 10           # remote window compared at each sweep point


def ranged(offset, length):
    req = urllib.request.Request(URL, headers={
        "User-Agent": "x", "Range": f"bytes={offset}-{offset + length - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def zero_scan(path):
    """Blocks that are almost entirely 0x00 -- i.e. never written."""
    holes, total = [], os.path.getsize(path)
    with open(path, "rb") as f:
        off, run_start = 0, None
        while True:
            b = f.read(BLOCK)
            if not b:
                break
            empty = b.count(0) >= ZERO_FRAC * len(b)
            if empty and run_start is None:
                run_start = off
            elif not empty and run_start is not None:
                holes.append((run_start, off - run_start))
                run_start = None
            off += len(b)
        if run_start is not None:
            holes.append((run_start, off - run_start))
    print(f"[loc] zero scan over {total / 1e9:.2f} GB: {len(holes)} hole region(s)")
    for start, size in holes:
        print(f"[loc]   hole at {start:>12,} ({start / 1e9:.3f} GB) "
              f"length {size:,} ({size / 1e6:.1f} MB)")
    return holes, total


def remote_sweep(path, total, step=SWEEP, win=WIN):
    """Compare local bytes against the remote at regular offsets."""
    bad = []
    points = list(range(0, max(1, total - win), step))
    print(f"[loc] remote sweep: {len(points)} points, {win // 1024} KB each")
    with open(path, "rb") as f:
        for i, off in enumerate(points):
            f.seek(off)
            local = f.read(win)
            try:
                remote = ranged(off, win)
            except Exception as e:  # noqa: BLE001 -- a probe failure is not a verdict
                print(f"[loc]   {off:,}: probe failed {type(e).__name__}")
                continue
            if local != remote:
                first = next((k for k in range(min(len(local), len(remote)))
                              if local[k] != remote[k]), None)
                bad.append(off if first is None else off + first)
                print(f"[loc]   MISMATCH at {off:,} ({off / 1e9:.3f} GB)"
                      f"{'' if first is None else f' first bad byte +{first}'}: "
                      f"local {local[:16].hex()} vs remote {remote[:16].hex()}")
            if i % 20 == 0:
                print(f"[loc]   ... {i}/{len(points)}", flush=True)
    print(f"[loc] sweep found {len(bad)} mismatching point(s)")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remote", action="store_true", help="also probe the CDN")
    ap.add_argument("--step", type=int, default=SWEEP)
    args = ap.parse_args()

    holes, total = zero_scan(DEST)
    covered = sum(s for _, s in holes)
    print(f"[loc] hole bytes: {covered:,} ({100 * covered / total:.2f}% of the file)")

    if args.remote and holes:
        print("\n[loc] confirming a hole against the remote:")
        start, size = max(holes, key=lambda h: h[1])
        probe = min(size, WIN)
        with open(DEST, "rb") as f:
            f.seek(start)
            local = f.read(probe)
        remote = ranged(start, probe)
        print(f"[loc]   at {start:,}: local all-zero={not any(local)} "
              f"remote all-zero={not any(remote)} match={local == remote}")

    if args.remote:
        print()
        remote_sweep(DEST, total, step=args.step)


if __name__ == "__main__":
    main()
