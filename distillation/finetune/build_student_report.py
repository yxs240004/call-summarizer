"""Build student_report.html: the distilled student's summaries beside its teacher's.

summaries_report.html already shows the reference and the four teacher-side arms on all 144
calls, but not a single line of the distilled student -- the thing this project was built to
produce. This reads the same eval set and score files the rest of the project uses and emits a
sibling page whose whole job is the mission's own question: read the 2B's summary, and say
whether you would ship it.

Four student arms, in the order a reader needs them: the student as trained (16-bit inside
transformers), those *same* 16-bit weights served through Ollama at the same seed -- the
control that separates precision from engine and draw -- the shipped Q4_K_M artifact, and
that same artifact decoded greedily -- the control that separates decoding from the weights.
The third is the one the mission is about; the second makes its 0.0162 attributable, and the
fourth is the only one that is a config change rather than a build.

Every arm's per-call row comes from the sibling scorer, so the scoreboard on each card is a
measurement rather than a restatement. The reference row carries no scores by construction.

    python -I build_student_report.py            # -> ../student_report.html
    python -I build_student_report.py --out x.html
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, os.pardir, "call-summarizer", "eval")
EVAL_SET = os.path.join(EVAL, "eval_set.jsonl")
SCORES = os.path.join(EVAL, "scores.json")           # the four teacher-side arms
SCORES_STU = os.path.join(HERE, "scores_q4km.json")   # blend, q4km
SCORES_F16 = os.path.join(HERE, "scores_f16.json")    # the precision control: 16-bit in Ollama
SCORES_GREEDY = os.path.join(HERE, "scores_greedy.json")  # the decoding control: greedy vs sampled
BLEND = os.path.join(HERE, "student-samples", "blend.jsonl")
Q4KM = os.path.join(HERE, "local_samples.jsonl")
F16 = os.path.join(HERE, "local_samples_f16.jsonl")
GREEDY = os.path.join(HERE, "local_samples_greedy.jsonl")
SIBLING_REPORT = os.path.join(ROOT, "summaries_report.html")  # for call metadata

# arm key, label, group, source-of-scores
ARMS = [
    ("reference",      "Reference",              "Reference",   "human",   None),
    ("qwen8b_lora",    "Teacher &middot; 8B tuned",   "Teacher",  "model",  "scores"),
    ("qwen4b",         "Baseline &middot; 4B tuned",  "4B tuned", "model",  "scores"),
    ("qwen8b_honest",  "Baseline &middot; 8B untuned", "8B untuned", "model", "scores"),
    ("student_blend",  "Student &middot; 16-bit",     "Student 16-bit", "model", "stu"),
    ("student_f16",    "Student &middot; 16-bit, same engine", "Student 16-bit Ollama", "model", "f16"),
    ("student_q4km",   "Student &middot; shipped 4-bit", "Student shipped", "model", "stu"),
    ("student_greedy", "Student &middot; shipped 4-bit, decoded greedily",
                       "Student greedy", "model", "greedy"),
]
CORE = {"reference", "qwen8b_lora", "student_blend", "student_f16", "student_q4km",
        "student_greedy"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_call(rows, key="call_id"):
    return {r[key]: r for r in rows}


def sibling_meta():
    """The existing report already carries per-call title/date/dur/dir/outcome."""
    meta = {}
    if not os.path.exists(SIBLING_REPORT):
        return meta
    with open(SIBLING_REPORT, encoding="utf-8") as f:
        blob = f.read()
    i = blob.find('<script type="application/json" id="data">')
    if i < 0:
        return meta
    j = blob.index(">", i) + 1
    k = blob.index("</script>", j)
    try:
        for rec in json.loads(blob[j:k]):
            meta[rec["id"]] = {x: rec.get(x) for x in
                               ("title", "date", "dur", "dir", "outcome")}
    except json.JSONDecodeError:
        pass
    return meta


def scores_for(path, arm_key):
    """Per-call rows for one arm, keyed by call_id, as {call_id: row}."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    rows = blob.get("rows", {}).get(arm_key)
    if not rows:
        return {}
    return {r["call_id"]: r for r in rows}


def per_variant(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("per_variant", {})


def mean(d, key):
    if key not in d:
        return None
    v = d[key]
    return v["mean"] if isinstance(v, dict) else v


def ci(d, key):
    if key not in d:
        return None
    v = d[key]
    return v.get("ci95") if isinstance(v, dict) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "student_report.html"))
    args = ap.parse_args()

    ev = load_jsonl(EVAL_SET)
    meta = sibling_meta()
    blend = index_by_call(load_jsonl(BLEND))
    q4km = index_by_call(load_jsonl(Q4KM))
    f16 = index_by_call(load_jsonl(F16))
    greedy = index_by_call(load_jsonl(GREEDY))

    sch = {a: scores_for(SCORES, a) for a in ("qwen8b_lora", "qwen4b", "qwen8b_honest")}
    st = {a: scores_for(SCORES_STU, a) for a in ("student_blend", "student_q4km")}
    sf = {a: scores_for(SCORES_F16, a) for a in ("student_f16",)}
    sg = {a: scores_for(SCORES_GREEDY, a) for a in ("student_greedy",)}
    scored_arms = list(sch.items()) + list(st.items()) + list(sf.items()) + list(sg.items())

    # Fail loudly rather than emit a page with silently missing arms.
    problems = []
    if len(ev) != 144:
        problems.append(f"eval set has {len(ev)} calls, expected 144")
    for a, d in scored_arms:
        if not d:
            problems.append(f"no per-call scores for {a}")
        elif len(d) != 144:
            problems.append(f"{a}: {len(d)} scored calls, expected 144")
    if not blend:
        problems.append("no 16-bit student samples")
    if not q4km:
        problems.append("no deployed student samples")
    if not f16:
        problems.append("no same-engine 16-bit control samples")
    if not greedy:
        problems.append("no greedy-decoding control samples")
    if problems:
        print("[report] refusing to build:")
        for p in problems:
            print("   -", p)
        return 1

    ev_ids = [r["call_id"] for r in ev]
    extra = set(blend) - set(ev_ids)
    if extra:
        print(f"[report] note: {len(extra)} sample ids outside the eval set, ignored")

    calls = []
    for r in ev:
        cid = r["call_id"]
        texts = {
            "reference": r.get("reference_summary") or "",
            "qwen8b_lora": (r.get("variants") or {}).get("qwen8b_lora") or "",
            "qwen4b": (r.get("variants") or {}).get("qwen4b") or "",
            "qwen8b_honest": (r.get("variants") or {}).get("qwen8b_honest") or "",
            "student_blend": (blend.get(cid) or {}).get("summary") or "",
            "student_f16": (f16.get(cid) or {}).get("summary") or "",
            "student_q4km": (q4km.get(cid) or {}).get("summary") or "",
            "student_greedy": (greedy.get(cid) or {}).get("summary") or "",
        }
        sc = {}
        for a, d in scored_arms:
            row = d.get(cid)
            if row:
                sc[a] = {
                    "rouge1": row["scores"]["rouge1"],
                    "rougeL": row["scores"]["rougeL"],
                    "words": row["words"],
                    "names": row["ungrounded_names"],
                    "nums": row["ungrounded_numbers"],
                    "dup": round(row["duplicate_rate"], 3),
                    "copy": round(row["copy_rate"], 3),
                    "empty": row["empty"],
                    "mid": row["mid_sentence"],
                }
        m = meta.get(cid, {})
        lat = q4km.get(cid) or {}
        calls.append({
            "id": cid,
            "title": m.get("title") or (blend.get(cid) or {}).get("title") or "",
            "date": m.get("date"), "dur": m.get("dur"),
            "dir": m.get("dir"), "outcome": m.get("outcome"),
            "transcript": r.get("cleaned_transcript") or "",
            "texts": texts,
            "scores": sc,
            "wall": lat.get("wall_s"), "tok": lat.get("output_tokens"),
        })

    pv = {}
    pv.update(per_variant(SCORES))
    pv.update(per_variant(SCORES_STU))
    pv.update(per_variant(SCORES_F16))
    pv.update(per_variant(SCORES_GREEDY))

    stats = {}
    for key, _lab, _s, _g, _src in ARMS:
        d = pv.get(key)
        if not d:
            continue
        stats[key] = {
            "rouge1": mean(d, "rouge1"), "ci": ci(d, "rouge1"),
            "rougeL": mean(d, "rougeL"),
            "words": d.get("output_words"),
            "names": d.get("ungrounded_names"),
            "pct": d.get("pct_calls_with_ungrounded"),
            "copy": d.get("mean_copy_rate"),
            "dup": d.get("mean_duplicate_rate"),
            "n": d.get("n"),
        }

    arms = [{"key": k, "label": lab, "short": s, "group": g, "scored": src is not None,
             "core": k in CORE} for k, lab, s, g, src in ARMS]

    payload = {"arms": arms, "calls": calls, "stats": stats}

    esc = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    doc = (HEAD.replace("__TITLE__", "Student vs Teacher")
               .replace("__CSS__", CSS)
               .replace("__DATA__", esc)
               .replace("__JS__", JS))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)

    n = len(calls)
    print(f"[report] {n} calls -> {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    for k, s in stats.items():
        c = s["ci"] or [0, 0]
        print(f"   {k:<16} R-1 {s['rouge1']:.4f} [{c[0]:.4f},{c[1]:.4f}]  "
              f"words {s['words']:.1f}  ungrdNm {s['names']}  pct {s['pct']}%")
    return 0


HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
__CSS__
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">Knowledge distillation &middot; side-by-side reading</p>
  <h1>The distilled student, against its teacher</h1>
  <p class="lede">
    The same 144 sales calls, summarised by a 2B student and by the 8B teacher it was trained
    to imitate. Read down each card: the reference the humans wrote, then the teacher, then the
    student as it was trained, then those same 16-bit weights served through the engine it
    actually ships on, then the student quantized to 4 bits, then that same 4-bit artifact with
    decoding switched from sampling to greedy. Then answer the only question the
    mission set &mdash; <em>would you send this one?</em>
  </p>
  <div class="tiles" id="tiles"></div>
  <div class="toolbar">
    <div class="field"><input id="q" type="search" placeholder="Search call id, title, or summary text&hellip;" aria-label="Search calls"></div>
    <label class="selwrap"><span class="selwrap__lab">Sort</span>
      <select id="sort" class="sel">
        <option value="worst">Worst for the shipped model first</option>
        <option value="best">Best for the shipped model first</option>
        <option value="gap">Biggest gap to the teacher first</option>
        <option value="omission">Shortest shipped summary first</option>
        <option value="names">Most ungrounded names first</option>
        <option value="id">Call id</option>
      </select>
    </label>
    <button class="seg-btn" id="base" aria-pressed="false">Show baselines</button>
    <button class="seg-btn" id="marksonly" aria-pressed="false">Only my marked calls</button>
    <span class="count" id="count"></span>
  </div>
  <div class="tally" id="tally"></div>
  <div class="list" id="list"></div>
  <div class="empty" id="empty" hidden>No call matches that filter.</div>
  <footer>
    <p>
      <strong>What the numbers are.</strong> ROUGE-1 and ROUGE-L are computed by the same scorer,
      against the same references, that produced every other figure in this project
      (<code>call-summarizer/eval/score.py</code>). &ldquo;Names&rdquo; and &ldquo;nums&rdquo; count
      proper nouns and figures in a summary that do not appear anywhere in that call&rsquo;s
      transcript &mdash; the invention measure. Words is the summary length.
    </p>
    <p>
      <strong>What the comparison moves, and what that costs.</strong> The 16-bit student ran
      inside transformers from an unseeded draw; the shipped 4-bit student is llama.cpp through
      Ollama at seed 42. Both decode at temperature 0.7, top_p 0.9 and repetition penalty 1.05,
      so <em>within that pair</em> sampling is not a difference &mdash; precision, engine and draw
      are. Serving the
      <em>16-bit</em> weights through the same engine at the same seed is that separation, and it
      is the fourth arm above: it scores <strong>0.4447</strong> against the shipped model's
      <strong>0.4285</strong>. A paired bootstrap over the same 144 calls calls that
      <strong>+0.0162 [+0.0057, +0.0266]</strong> &mdash; real. The engine-and-draw term on its own
      is <strong>&minus;0.0048 [&minus;0.0164, +0.0067]</strong>, a tie, and the two terms sum to
      the +0.0114 that step 8 first published. So <strong>the cost of shipping 4-bit is the
      quantization, and it is 0.0162 rather than 0.0114</strong> &mdash; the earlier figure was
      diluted by an engine term pulling the other way inside its own noise. A caveat worth
      carrying: seed 42 fixes the <em>draw</em>, not the output. Different logits mean the token
      streams still diverge, so the seed removes the draw as a nuisance variable rather than
      making the two arms comparable token for token.
    </p>
    <p>
      <strong>The one thing here that is a config change rather than a build.</strong> The last
      arm is the same 1.27 GB artifact, the same engine and the same seed, with decoding switched
      from sampling to greedy &mdash; so decoding is the only variable, and it is a flag rather
      than a retraining run. It cuts invented names from <strong>37 to 24</strong> and invented
      numbers from <strong>105 to 76</strong> at an <em>identical length</em> (108.7 words against
      108.0), which rules out the boring explanation that a shorter summary has fewer chances to
      invent. What it does <em>not</em> do is improve ROUGE in any way this run can detect:
      ROUGE-2 moves <strong>+0.0119 [+0.0035, +0.0193]</strong>, real, while ROUGE-1 and ROUGE-L
      both tie. The honest sentence is therefore <strong>&ldquo;greedy is not worse on
      ROUGE&rdquo;</strong>, with the grounding counts &mdash; which carry no paired test &mdash;
      doing the rest of the work. Two costs travel with it: one of the 144 greedy calls ran into
      the 900-token cap and stopped mid-sentence where the sampled run had none, and it takes
      1.16&times; the wall time. It is also a <em>net</em> reduction rather than a per-call
      improvement: sampling invents more names on 19 calls and greedy on 8, so a given card can
      go either way. Mark a few calls and read both columns before deciding.
    </p>
    <p>
      <strong>Your marks stay in this browser</strong>, on this machine only. They are not sent
      anywhere and are not part of the model's evaluation.
    </p>
  </footer>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
__JS__
</script>
</body>
</html>
"""

CSS = r"""
  /* Tokens shared with summaries_report.html and architecture.html, so the three
     pages read as one system. accent = the shipped student, ink = the teacher,
     warn = a baseline. No other element borrows those hues. */
  :root{
    --bg:#f3f7f7; --surface:#fff; --surface-2:#eaf1f1;
    --ink:#0d1719; --ink-2:#3c4e53; --muted:#6a7f84;
    --rule:#d6e0e1; --rule-2:#bfced0;
    --accent:#0b7168; --accent-bg:#e0f0ed;
    --warn:#8d5200; --warn-bg:#fbf0dc; --warn-line:#e5c99a;
    --stop:#8f1d1d; --stop-bg:#fbeaea;
    --display:'Archivo','Helvetica Neue',Helvetica,Arial,sans-serif;
    --body:'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --bg:#0b1214; --surface:#141d20; --surface-2:#1a2528;
      --ink:#e7efef; --ink-2:#b4c4c7; --muted:#8ba0a5;
      --rule:#243336; --rule-2:#33464b;
      --accent:#4dc0b1; --accent-bg:#0f2a28;
      --warn:#dfa75b; --warn-bg:#2a2013; --warn-line:#5b4720;
      --stop:#e08b8b; --stop-bg:#2b1414;
      color-scheme:dark;
    }
  }
  :root[data-theme="dark"]{
    --bg:#0b1214; --surface:#141d20; --surface-2:#1a2528;
    --ink:#e7efef; --ink-2:#b4c4c7; --muted:#8ba0a5;
    --rule:#243336; --rule-2:#33464b;
    --accent:#4dc0b1; --accent-bg:#0f2a28;
    --warn:#dfa75b; --warn-bg:#2a2013; --warn-line:#5b4720;
    --stop:#e08b8b; --stop-bg:#2b1414;
    color-scheme:dark;
  }

  *{box-sizing:border-box}
  [hidden]{display:none !important}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:var(--body);font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1080px;margin-inline:auto;padding-inline:20px;padding-block:38px 72px}
  .wrap > *{min-width:0}

  .eyebrow{font-family:var(--mono);font-size:11px;font-weight:500;letter-spacing:.1em;
    text-transform:uppercase;color:var(--muted);margin:0 0 12px}
  h1{font-family:var(--display);font-weight:700;font-size:clamp(29px,4.4vw,42px);
    line-height:1.06;letter-spacing:-.02em;text-wrap:balance;margin:0 0 12px}
  .lede{font-size:17px;color:var(--ink-2);max-width:68ch;margin:0}

  /* headline tiles */
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:1px;
    background:var(--rule);border:1px solid var(--rule);border-radius:10px;
    overflow:hidden;margin:30px 0 0}
  .tile{background:var(--surface);padding:15px 17px 13px}
  .tile .k{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;
    color:var(--muted);display:block;margin-bottom:7px}
  .tile .v{font-family:var(--display);font-weight:700;font-size:22px;line-height:1.15;
    letter-spacing:-.01em;font-variant-numeric:tabular-nums;display:block}
  .tile .n{font-size:12.5px;color:var(--muted);display:block;margin-top:5px}

  /* toolbar */
  .toolbar{position:sticky;top:env(safe-area-inset-top,0px);z-index:6;
    display:flex;flex-wrap:wrap;gap:10px 12px;align-items:center;
    margin:26px 0 0;padding:12px 0 13px;background:var(--bg);border-bottom:1px solid var(--rule)}
  .field{position:relative;flex:1 1 230px;min-width:0}
  .field input{width:100%;font-family:var(--body);font-size:14.5px;color:var(--ink);
    background:var(--surface);border:1px solid var(--rule-2);border-radius:8px;padding:9px 12px}
  .field input:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:var(--accent)}
  .selwrap{display:flex;align-items:center;gap:8px}
  .selwrap__lab{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
    text-transform:uppercase;color:var(--muted)}
  .sel{font-family:var(--mono);font-size:12px;color:var(--ink-2);background:var(--surface);
    border:1px solid var(--rule-2);border-radius:8px;padding:9px 10px;max-width:100%}
  .sel:focus{outline:2px solid var(--accent);outline-offset:1px}
  .seg-btn{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
    color:var(--muted);background:var(--surface);border:1px solid var(--rule-2);
    border-radius:8px;padding:9px 12px;cursor:pointer;white-space:nowrap}
  .seg-btn[aria-pressed="true"]{background:var(--surface-2);color:var(--ink);font-weight:600}
  .seg-btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
  .count{font-family:var(--mono);font-size:12px;color:var(--muted);margin-left:auto;white-space:nowrap}

  /* your tally */
  .tally{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline;
    margin:14px 0 0;font-family:var(--mono);font-size:12px;color:var(--muted)}
  .tally b{font-family:var(--display);font-weight:700;font-size:16px;color:var(--ink);
    font-variant-numeric:tabular-nums}
  .tally .ship b{color:var(--accent)}
  .tally .no b{color:var(--stop)}
  .tally button{font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;
    text-transform:uppercase;color:var(--muted);background:transparent;
    border:1px solid var(--rule-2);border-radius:6px;padding:3px 8px;cursor:pointer}
  .tally button:hover{color:var(--ink);border-color:var(--muted)}

  /* cards */
  .list{display:grid;gap:18px;margin-top:20px}
  .card{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
    overflow:hidden;scroll-margin-top:96px}
  .card[data-mark="ship"]{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}
  .card[data-mark="no"]{border-color:var(--stop);box-shadow:inset 3px 0 0 var(--stop)}
  .card-h{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px;
    padding:15px 18px 13px;border-bottom:1px solid var(--rule)}
  .card-h .t{font-family:var(--display);font-weight:600;font-size:16.5px;
    letter-spacing:-.005em;margin:0}
  .card-h .m{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
  .chip{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
    border:1px solid var(--rule-2);border-radius:99px;padding:2px 8px;color:var(--muted)}
  .chip--flag{border-color:var(--warn-line);background:var(--warn-bg);color:var(--warn)}

  /* scoreboard */
  .board{margin:0;padding:12px 18px 14px;border-bottom:1px solid var(--rule);
    background:var(--surface-2)}
  .board table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px}
  .board th{text-align:right;font-weight:500;font-size:9.5px;letter-spacing:.08em;
    text-transform:uppercase;color:var(--muted);padding:0 0 6px;white-space:nowrap}
  .board th:first-child,.board td:first-child{text-align:left}
  .board td{text-align:right;padding:3px 0;font-variant-numeric:tabular-nums;color:var(--ink-2);
    white-space:nowrap}
  .board tr.is-ref td{color:var(--muted)}
  .board tr.is-teacher td:first-child{color:var(--ink);font-weight:500}
  .board tr.is-ship td:first-child{color:var(--accent);font-weight:600}
  .board tr.is-base td:first-child{color:var(--warn)}
  .board .best{color:var(--ink);font-weight:600}
  .board .bad{color:var(--stop);font-weight:600}
  .board .dash{color:var(--rule-2)}
  .board-wrap{overflow-x:auto}

  /* arms */
  .arm{padding:14px 18px 16px;border-bottom:1px solid var(--rule)}
  .arm:last-of-type{border-bottom:0}
  .arm .lab{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px;margin-bottom:7px}
  .arm .who{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
    text-transform:uppercase;font-weight:600;color:var(--muted)}
  .arm--reference .who{color:var(--muted)}
  .arm--teacher .who{color:var(--ink)}
  .arm--baseline .who{color:var(--warn)}
  .arm--trained .who{color:var(--ink-2)}
  .arm--ship .who{color:var(--accent)}
  .arm .n{font-family:var(--mono);font-size:10.5px;color:var(--muted)}
  .arm p{margin:0;font-size:15.5px;line-height:1.64;color:var(--ink)}
  .arm--reference p{font-size:16px;border-left:3px solid var(--rule-2);padding-left:14px}
  .arm--ship{background:var(--accent-bg)}
  .arm--ship p{font-size:16px}
  .copy{margin-left:auto;font-family:var(--mono);font-size:10px;letter-spacing:.07em;
    text-transform:uppercase;color:var(--muted);background:transparent;
    border:1px solid var(--rule-2);border-radius:6px;padding:2px 7px;cursor:pointer}
  .copy:hover{color:var(--ink);border-color:var(--muted)}
  .copy:focus-visible{outline:2px solid var(--accent);outline-offset:1px}

  /* your verdict */
  .verdict{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
    padding:12px 18px;border-top:1px solid var(--rule);background:var(--surface)}
  .verdict .vl{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;
    text-transform:uppercase;color:var(--muted);margin-right:2px}
  .verdict button{font-family:var(--mono);font-size:11px;letter-spacing:.05em;
    color:var(--ink-2);background:var(--bg);border:1px solid var(--rule-2);
    border-radius:7px;padding:6px 11px;cursor:pointer}
  .verdict button:hover{border-color:var(--muted)}
  .verdict button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
  .verdict button[aria-pressed="true"]{font-weight:600}
  .verdict button[data-v="ship"][aria-pressed="true"]{background:var(--accent);
    border-color:var(--accent);color:var(--surface)}
  .verdict button[data-v="no"][aria-pressed="true"]{background:var(--stop);
    border-color:var(--stop);color:var(--surface)}
  .verdict button[data-v="unsure"][aria-pressed="true"]{background:var(--warn);
    border-color:var(--warn);color:var(--surface)}

  details{border-top:1px solid var(--rule)}
  details > summary{font-family:var(--mono);font-size:11px;letter-spacing:.07em;
    text-transform:uppercase;color:var(--muted);padding:11px 18px;cursor:pointer;list-style:none}
  details > summary::-webkit-details-marker{display:none}
  details > summary::before{content:"+";display:inline-block;width:1.4em;color:var(--muted)}
  details[open] > summary::before{content:"\2212"}
  details > summary:hover{color:var(--ink)}
  details > summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
  details .dbody{padding:0 18px 18px}
  details pre{margin:0;font-family:var(--mono);font-size:12.5px;line-height:1.72;
    color:var(--ink-2);white-space:pre-wrap;word-break:break-word;max-height:340px;
    overflow:auto;background:var(--bg);border:1px solid var(--rule);border-radius:8px;
    padding:12px 14px}

  .empty{text-align:center;color:var(--muted);font-size:15px;padding:56px 20px}
  footer{margin-top:52px;border-top:1px solid var(--rule);padding-top:20px;
    font-size:13px;color:var(--muted);max-width:82ch}
  footer p{margin:0 0 10px;font-size:13px}
  footer code{font-family:var(--mono);font-size:12px}

  @media (max-width:640px){
    .count{margin-left:0;width:100%}
    .arm{padding:13px 15px 15px}
    .board,.card-h,.verdict,details > summary{padding-inline:15px}
  }
"""

JS = r"""
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("data").textContent);
  var ARMS = D.arms, CALLS = D.calls, STATS = D.stats;
  var list = document.getElementById("list");
  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  };
  var num3 = function (x) { return x == null ? "—" : x.toFixed(3); };
  var f = function (x, d) { return x == null ? "—" : x.toFixed(d == null ? 1 : d); };

  /* ---- marks: per-viewer, best-effort, never load-bearing ---- */
  var store = (function () {
    try {
      var k = "distill.marks.v1";
      localStorage.setItem(k, "1"); localStorage.removeItem(k);
      return window.localStorage;
    } catch (e) { return null; }
  })();
  var KEY = "distill.student.marks.v1";
  var marks = {};
  try { marks = JSON.parse((store && store.getItem(KEY)) || "{}") || {}; }
  catch (e) { marks = {}; }
  function saveMarks() {
    try { if (store) store.setItem(KEY, JSON.stringify(marks)); } catch (e) {}
  }
  function markOf(id) { return marks[id] || null; }
  function setMark(id, v) {
    if (v === null) { delete marks[id]; } else { marks[id] = v; }
    saveMarks();
  }

  function tile(k, v, n) {
    return '<div class="tile"><span class="k">' + esc(k) + '</span>' +
           '<span class="v">' + v + '</span><span class="n">' + esc(n) + '</span></div>';
  }
  function renderTiles() {
    var t = STATS, ship = t.student_q4km, tea = t.qwen8b_lora, b16 = t.student_blend,
        f16s = t.student_f16, b4 = t.qwen4b, b8u = t.qwen8b_honest, grd = t.student_greedy;
    var h = "";
    if (ship && tea) {
      h += tile("Shipped vs teacher", (100 * ship.rouge1 / tea.rouge1).toFixed(0) + "%",
                "0.4285 against " + tea.rouge1.toFixed(4) + " ROUGE-1");
    }
    if (ship && f16s) {
      h += tile("Shipped vs its 16-bit self", (100 * ship.rouge1 / f16s.rouge1).toFixed(0) + "%",
                "same engine and seed: 0.4285 against " + f16s.rouge1.toFixed(4) +
                "; in transformers, 0.4399");
    } else if (ship && b16) {
      h += tile("Shipped vs its 16-bit self", (100 * ship.rouge1 / b16.rouge1).toFixed(0) + "%",
                "a gap of " + (b16.rouge1 - ship.rouge1).toFixed(4));
    }
    if (ship && f16s) {
      h += tile("What 4 bits cost", "+" + (f16s.rouge1 - ship.rouge1).toFixed(4),
                "ROUGE-1, precision alone; engine and draw cost nothing");
    }
    if (b16 && b4 && b8u) {
      h += tile("2B against a 4B fine-tune",
                (b16.rouge1 >= b4.rouge1 ? "+" : "") + (b16.rouge1 - b4.rouge1).toFixed(4),
                "16-bit student " + b16.rouge1.toFixed(4) + " vs 4B " + b4.rouge1.toFixed(4) +
                ", and the untuned 8B is " + b8u.rouge1.toFixed(4));
    }
    if (ship) {
      h += tile("Cost per call", f(16.0, 1) + " s",
                "median, CPU only; " + ship.words.toFixed(0) + " words out");
      h += tile("On disk", "1.27 GB", "quantized; the 16-bit weights are 3.78 GB");
      h += tile("Invented names", String(ship.names),
                "across " + ship.pct.toFixed(0) + "% of calls — diffuse, not a runaway loop");
    }
    if (grd && ship && grd.names < ship.names) {
      h += tile("Invented names, decoded greedily", String(grd.names),
                "against " + ship.names + " when sampling; ROUGE-1 moves " +
                (grd.rouge1 >= ship.rouge1 ? "+" : "") + (grd.rouge1 - ship.rouge1).toFixed(4) +
                ", a tie — one flag, no retraining");
    }
    document.getElementById("tiles").innerHTML = h;
  }

  /* `best` is computed once over every scored arm, not just the ones on screen, so
     toggling the baselines never moves the marker. */
  function scoreCells(c, arm, best) {
    if (!arm.scored) {
      return '<td class="dash">—</td><td class="dash">—</td>' +
             '<td>' + (c.texts.reference ? c.texts.reference.trim().split(/\s+/).length : "—") +
             '</td><td class="dash">—</td><td class="dash">—</td>';
    }
    var s = c.scores[arm.key];
    if (!s) {
      return '<td class="dash">—</td><td class="dash">—</td>' +
             '<td class="dash">—</td><td class="dash">—</td><td class="dash">—</td>';
    }
    return '<td class="' + (best >= 0 && s.rouge1 === best ? "best" : "") + '">' +
             num3(s.rouge1) + '</td>' +
           '<td>' + num3(s.rougeL) + '</td>' +
           '<td>' + s.words + '</td>' +
           '<td class="' + (s.names >= 4 ? "bad" : "") + '">' + s.names + '</td>' +
           '<td class="' + (s.nums >= 4 ? "bad" : "") + '">' + s.nums + '</td>';
  }

  function bestOf(c) {
    var b = -1;
    ARMS.forEach(function (a) {
      if (!a.scored) return;
      var s = c.scores[a.key];
      if (s && s.rouge1 > b) b = s.rouge1;
    });
    return b;
  }

  function tallyHtml() {
    var marked = 0, ship = 0, no = 0, un = 0;
    CALLS.forEach(function (c) {
      var v = markOf(c.id);
      if (!v) return;
      marked++; if (v === "ship") ship++; else if (v === "no") no++; else un++;
    });
    return '<span class="ship">would ship <b>' + ship + '</b></span>' +
           '<span class="no">would not <b>' + no + '</b></span>' +
           '<span>unsure <b>' + un + '</b></span>' +
           '<span>of <b>' + marked + '</b> marked, ' + CALLS.length + ' calls</span>' +
           (marked ? '<button id="clearmarks">Clear my marks</button>' : '');
  }

  function render() {
    var q = document.getElementById("q").value.trim().toLowerCase();
    var sort = document.getElementById("sort").value;
    var showBase = document.getElementById("base").getAttribute("aria-pressed") === "true";
    var markedOnly = document.getElementById("marksonly").getAttribute("aria-pressed") === "true";

    var arms = ARMS.filter(function (a) { return showBase || a.core; });

    var rows = CALLS.filter(function (c) {
      if (markedOnly && !markOf(c.id)) return false;
      if (!q) return true;
      if (c.id.toLowerCase().indexOf(q) >= 0) return true;
      if ((c.title || "").toLowerCase().indexOf(q) >= 0) return true;
      for (var k in c.texts) {
        if ((c.texts[k] || "").toLowerCase().indexOf(q) >= 0) return true;
      }
      return false;
    });

    var s1 = function (c, k) {
      var s = c.scores[k]; return s ? s.rouge1 : -1;
    };
    var sorted = rows.slice().sort(function (a, b) {
      if (sort === "worst") return s1(a, "student_q4km") - s1(b, "student_q4km");
      if (sort === "best") return s1(b, "student_q4km") - s1(a, "student_q4km");
      if (sort === "gap") {
        var ga = s1(a, "qwen8b_lora") - s1(a, "student_q4km");
        var gb = s1(b, "qwen8b_lora") - s1(b, "student_q4km");
        return gb - ga;
      }
      if (sort === "omission") {
        return (a.scores.student_q4km ? a.scores.student_q4km.words : 1e9) -
               (b.scores.student_q4km ? b.scores.student_q4km.words : 1e9);
      }
      if (sort === "names") {
        return (b.scores.student_q4km ? b.scores.student_q4km.names : 0) -
               (a.scores.student_q4km ? a.scores.student_q4km.names : 0);
      }
      return a.id < b.id ? -1 : 1;
    });

    var out = sorted.map(function (c) {
      var m = markOf(c.id);
      var best = bestOf(c);

      var meta = [c.id, c.date, c.dur ? Math.round(c.dur / 60) + " min" : null,
                  c.dir, c.outcome].filter(Boolean).map(esc).join(" &middot; ");

      var h = '<article class="card" id="c-' + esc(c.id) + '"' +
              (m ? ' data-mark="' + esc(m) + '"' : '') + '>';
      h += '<div class="card-h"><h2 class="t">' + esc(c.title || "Call " + c.id) + '</h2>' +
           '<span class="m">' + meta + '</span>';
      if (c.wall != null) h += '<span class="m">' + f(c.wall, 1) + ' s to serve</span>';
      h += '</div>';

      h += '<div class="board"><div class="board-wrap"><table><thead><tr>' +
           '<th>Arm</th><th>ROUGE-1</th><th>ROUGE-L</th><th>Words</th>' +
           '<th>Names</th><th>Nums</th></tr></thead><tbody>';
      arms.forEach(function (a) {
        var cls = a.key === "reference" ? "is-ref"
                : a.key === "qwen8b_lora" ? "is-teacher"
                : a.key === "student_q4km" ? "is-ship"
                : a.group === "model" && !a.core ? "is-base" : "";
        h += '<tr class="' + cls + '"><td>' + a.label.replace("&middot;", "·") + '</td>' +
             scoreCells(c, a, best) + '</tr>';
      });
      h += '</tbody></table></div></div>';

      arms.forEach(function (a) {
        var txt = c.texts[a.key] || "";
        var cls = a.key === "reference" ? "arm--reference"
                : a.key === "qwen8b_lora" ? "arm--teacher"
                : a.key === "student_q4km" ? "arm--ship"
                : a.group === "model" && !a.core ? "arm--baseline" : "arm--trained";
        var s = c.scores[a.key];
        var n = [];
        if (s) { n.push(s.words + " words"); if (s.names) n.push(s.names + " ungrounded names"); }
        if (!txt) n.push("no output recorded");
        h += '<div class="arm ' + cls + '"><div class="lab">' +
             '<span class="who">' + a.short + '</span>' +
             (n.length ? '<span class="n">' + esc(n.join(" · ")) + '</span>' : '') +
             '<button class="copy" data-copy="' + esc(c.id) + '|' + a.key +
             '">copy</button></div>' +
             '<p>' + (txt ? esc(txt) : '<em>nothing was generated for this arm</em>') + '</p></div>';
      });

      h += '<div class="verdict"><span class="vl">Would you send this one?</span>' +
           '<button data-mark="ship" data-id="' + esc(c.id) + '"' +
             (m === "ship" ? ' aria-pressed="true"' : '') + '>Ship it</button>' +
           '<button data-mark="no" data-id="' + esc(c.id) + '"' +
             (m === "no" ? ' aria-pressed="true"' : '') + '>Do not ship</button>' +
           '<button data-mark="unsure" data-id="' + esc(c.id) + '"' +
             (m === "unsure" ? ' aria-pressed="true"' : '') + '>Unsure</button></div>';

      h += '<details><summary>Transcript</summary><div class="dbody"><pre>' +
           esc(c.transcript) + '</pre></div></details>';
      return h + '</article>';
    }).join("");

    list.innerHTML = out;
    document.getElementById("empty").hidden = sorted.length > 0;
    document.getElementById("tally").innerHTML = tallyHtml();
    document.getElementById("count").textContent =
      sorted.length + " of " + CALLS.length + " calls";
  }

  list.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("button") : null;
    if (!b) return;
    if (b.classList.contains("copy")) {
      var parts = b.getAttribute("data-copy").split("|");
      var c = CALLS.filter(function (x) { return x.id === parts[0]; })[0];
      var txt = (c && c.texts[parts[1]]) || "";
      var done = function () { var o = b.textContent; b.textContent = "copied";
        setTimeout(function () { b.textContent = o; }, 1200); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(done, function () { b.textContent = "press copy"; });
      } else { b.textContent = "press copy"; }
      return;
    }
    if (b.hasAttribute("data-mark")) {
      var id = b.getAttribute("data-id"), v = b.getAttribute("data-mark");
      setMark(id, markOf(id) === v ? null : v);
      var card = document.getElementById("c-" + id);
      var now = markOf(id);
      if (card) { if (now) card.setAttribute("data-mark", now);
                  else card.removeAttribute("data-mark"); }
      var sib = b.parentNode.querySelectorAll("button[data-mark]");
      for (var i = 0; i < sib.length; i++) {
        if (sib[i].getAttribute("data-mark") === now) sib[i].setAttribute("aria-pressed", "true");
        else sib[i].removeAttribute("aria-pressed");
      }
      renderTallyOnly();
    }
  });

  function renderTallyOnly() {
    document.getElementById("tally").innerHTML = tallyHtml();
  }

  document.getElementById("tally").addEventListener("click", function (e) {
    var t = e.target.closest ? e.target.closest("button") : null;
    if (t && t.id === "clearmarks") { marks = {}; saveMarks(); render(); }
  });

  ["q", "sort"].forEach(function (id) {
    document.getElementById(id).addEventListener("input", render);
    document.getElementById(id).addEventListener("change", render);
  });
  ["base", "marksonly"].forEach(function (id) {
    document.getElementById(id).addEventListener("click", function () {
      var p = this.getAttribute("aria-pressed") === "true";
      this.setAttribute("aria-pressed", p ? "false" : "true");
      render();
    });
  });

  renderTiles();
  render();
})();
"""


if __name__ == "__main__":
    sys.exit(main())
