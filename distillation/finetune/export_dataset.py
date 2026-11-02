"""Export the 305 (transcript + metadata) -> summary training pairs from Supabase.

Uses urllib (NOT PowerShell / requests-with-browser-UA) per the known
sb_secret_ User-Agent trap. Writes finetune/dataset_raw.json — the untouched
records; formatting into model input happens later in build_dataset.py.
"""
import json
import os
import urllib.parse
import urllib.request

ENV_PATH = r"C:\Projects\call-summarizer\.env"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_raw.json")

SELECT = ",".join([
    "call_id", "title", "summmary", "duration_seconds", "deal_outcome",
    "stage_at_call_label", "attendees", "entities", "speaker_mapping",
    "speaker_labeled_transcript",
])


def load_env(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def fetch_all(url, key):
    """Page through PostgREST with a Range header until rows run out."""
    rows, page = [], 100
    offset = 0
    while True:
        q = urllib.parse.urlencode({
            "select": SELECT,
            "transcription_status": "eq.completed",
            "order": "call_id",
        })
        req = urllib.request.Request(f"{url}/rest/v1/calls_for_analysis?{q}")
        req.add_header("apikey", key)
        req.add_header("Accept", "application/json")
        req.add_header("Range-Unit", "items")
        req.add_header("Range", f"{offset}-{offset + page - 1}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows


def main():
    env = load_env(ENV_PATH)
    url, key = env["SUPABASE_URL"], env["SUPABASE_SECRET"]
    rows = fetch_all(url, key)
    print(f"fetched {len(rows)} completed rows")

    keep = []
    for r in rows:
        sl = r.get("speaker_labeled_transcript") or {}
        segments = sl.get("segments") or []
        keep.append({
            "call_id": r["call_id"],
            "title": r.get("title"),
            "duration_seconds": r.get("duration_seconds"),
            "deal_outcome": r.get("deal_outcome"),
            "stage_at_call_label": r.get("stage_at_call_label"),
            "attendees": r.get("attendees"),
            "entities": r.get("entities"),
            "speaker_mapping": r.get("speaker_mapping"),
            "target_summary": r.get("summmary"),
            "segments": [
                {"speaker": s.get("speaker"), "role": s.get("role"), "text": s.get("text")}
                for s in segments
            ],
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(keep)} records -> {OUT_PATH}")

    # quick size profile to inform the input-format / truncation design
    with_target = sum(1 for k in keep if k["target_summary"])
    with_segs = sum(1 for k in keep if k["segments"])
    chars = [sum(len(s["text"] or "") for s in k["segments"]) for k in keep if k["segments"]]
    chars.sort()
    if chars:
        n = len(chars)
        print(f"with target summary: {with_target} | with segments: {with_segs}")
        print("transcript chars: min=%d median=%d p90=%d max=%d" % (
            chars[0], chars[n // 2], chars[int(n * 0.9)], chars[-1]))


if __name__ == "__main__":
    main()
