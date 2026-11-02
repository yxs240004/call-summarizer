"""
Score every variant in the frozen eval set against the house reference.

Kept from the t5 harness: ROUGE-1/2/L with stemming, bootstrap 95% CI, and the
failure modes ROUGE hides -- repetition, verbatim copying from the input, empty
output. Two things are added, because ROUGE cannot see them and they are the
failure a small model actually exhibits:

  * grounding -- share of dollar amounts and proper nouns in the output that do
    not appear anywhere in the transcript the model read. That is fabrication.
  * paired deltas -- the same calls scored by two variants, so the difference
    between them gets its own CI. Comparing two means is not the same test.

Reads eval/eval_set.jsonl; writes eval/scores.json.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys

from rouge_score import rouge_scorer

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["rouge1", "rouge2", "rougeL"]
BOOTSTRAP = 2000
# the text each variant actually read, for a meaningful copy_rate
COPY_CORPUS = {
    "qwen8b_lora": "model_input",
    "qwen8b_honest": "model_input",
    "qwen4b": "model_input",
    "bart": "cleaned_transcript",
    # The distilled student read model_input too -- it is prompted from the same
    # infer_input.jsonl {system,user} pair as the Qwen baselines, and model_input is
    # byte-identical to that user field on all 144 calls. Additive only: every existing
    # variant keeps its entry, so the four baseline rows reproduce unchanged.
    "student_blend": "model_input",
    "student_control": "model_input",
    # student_q4km was absent from this table when step 8 scored the deployed artifact,
    # so it fell through to the cleaned_transcript default and reported 324 ungrounded
    # names where the matched corpus gives 37 -- a 9x inflation on the one metric that
    # catches what ROUGE cannot see, and it was published. Registering the name was the
    # fix for student_blend; that fix did not survive a *new* variant name, which is why
    # the lookup below now refuses to guess. See corpus_for().
    "student_q4km": "model_input",
    "student_f16": "model_input",
    "student_greedy": "model_input",
    "student_q8": "model_input",
    # The metadata A/B arms read different prompts by construction, so they must not
    # share model_input: grounding the `meta` arm against a prompt without its
    # `Prospect:` line would flag a summary that correctly repeats a name it was given.
    # make_student_eval.py --inputs attaches each arm's own prompt as input_<arm>.
    "student_plain": "input_plain",
    "student_meta": "input_meta",
}

# Per-run overrides from `--corpus variant=field`; see corpus_for().
CORPUS_OVERRIDE = {}


def corpus_for(variant, row):
    """The text this variant actually read, or stop rather than pick one.

    This used to be `COPY_CORPUS.get(variant, "cleaned_transcript")`. cleaned_transcript
    is raw ASR with anonymised SPEAKER_00/SPEAKER_01 labels, so the call participants'
    own names never appear in it, and every summary that names the person on the call --
    most of them, and what the house reference does too -- reads as invented. Scoring a
    Qwen-family variant against it inflates ungrounded_names about ninefold (413 against
    46 for student_blend; 324 against 37 for student_q4km). A missing entry is now a hard
    stop: on this metric a silent wrong answer is worse than a failed run.

    `--corpus variant=field` overrides the registry for one run. It exists for A/Bs whose
    arms read *different* prompts: grounding each arm against its own prompt makes the
    metric incomparable, because a fact removed from one arm's prompt (a deal stage, a
    prospect name) is then counted as invented when the model correctly infers it. Such a
    run should point both arms at the same, fullest factual record for the call.
    """
    field = CORPUS_OVERRIDE.get(variant)
    if field is None:
        try:
            field = COPY_CORPUS[variant]
        except KeyError:
            sys.exit(f"{variant}: not registered in COPY_CORPUS, so there is no record of "
                     f"which text it actually read. Add it there; do not let the scorer "
                     f"choose a corpus on your behalf.")
    return row[field]


# magnitude suffixes matter: "$84K" and "84000" are the same figure, and flagging
# the second as invented would make the fabrication number useless. Percentages are
# NOT scaled -- 1.5% stays 1.5. Word forms count too: the transcripts say "15 million"
# where the summaries write "$15,000,000", and without them that reads as invention.
NUMBER_RE = re.compile(
    r"(?<![\w.])(\$)?\s?(\d[\d,]*(?:\.\d+)?)(?:st|nd|rd|th)?\s?"
    r"(%|percent|hundred|thousand|million|billion|trillion|bn|mm|k|m|b)?(?![\w])", re.I)
MAGNITUDE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "trillion": 1e12, "hundred": 1e2,
}
PCT_SUFFIXES = ("%", "percent")

# These are sales CALLS, so most figures are spoken: "nine and a half percent",
# "five hundred thousand", "1 point five percent". A digit-only extractor reads
# every one of those as invention, which is the single biggest false-positive class
# in this metric. So spoken numbers get parsed too, and the two sets are unioned.
#
# Two residual false positives survive on purpose, both cheap to state and hard to
# fix: ordinals are matched ("18th" == "18"), but an ASR run-together time is not
# -- "1030 AM" in the transcript against "10:30 AM" in the summary reads as two
# invented figures. Audit a flagged figure before calling it a fabrication.
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9,
          "trillion": 1e12}
NUMWORD = set(UNITS) | set(TENS) | set(SCALES)
_ALT = lambda words: "|".join(sorted(words, key=len, reverse=True))
SPOKEN_RE = re.compile(
    r"(?<![\w.])(?:\d[\d,]*(?:\.\d+)?|" + _ALT(NUMWORD | {"half"}) + r")"
    r"(?:\s+(?:" + _ALT(NUMWORD | {"and", "a", "half", "point"}) + r"))*"
    r"\s*(?:percent|%)?", re.I)
CAPITALIZED_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
TRIM = '.,;:()"\''
# capitalized words that are not names; keeps the proper-noun proxy honest
STOPWORDS = {
    "the", "this", "that", "they", "there", "then", "these", "those", "and", "but",
    "for", "with", "was", "were", "will", "would", "what", "when", "where", "which",
    "who", "why", "how", "his", "her", "him", "she", "you", "your", "our", "its",
    "from", "into", "about", "after", "also", "just", "like", "made", "make", "more",
    "most", "new", "not", "now", "one", "only", "other", "out", "over", "said", "same",
    "some", "still", "such", "than", "them", "very", "well", "been", "both", "call",
    "could", "did", "does", "each", "even", "every", "had", "has", "have", "here",
    "rep", "prospect", "next", "steps", "outcome", "signal", "deal", "summary",
}


def ngrams(text, n=4):
    toks = text.lower().split()
    return [tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))]


def duplicate_rate(text, n=4):
    grams = ngrams(text, n)
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def copy_rate(generated, source, n=4):
    """Share of the output's 4-grams lifted verbatim from the text it read."""
    grams = ngrams(generated, n)
    if not grams:
        return 0.0
    src = set(ngrams(source, n))
    return sum(1 for g in grams if g in src) / len(grams)


def _eval_run(text):
    """Value of a spoken number phrase: 'nine and a half' -> 9.5, '1 point five' -> 1.5.

    'a' and 'and' carry no value and are dropped -- 'a hundred thousand' works
    because a bare scale word implies one. Unknown words are ignored rather than
    raising, since the run can pick up a trailing 'percent'.
    """
    toks = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower())
    if "point" in toks:
        i = toks.index("point")
        head, tail = toks[:i], toks[i + 1:]
    else:
        head, tail = toks, []

    total = current = 0.0
    for t in head:
        if t in ("and", "a", "point"):
            continue
        if t == "half":
            current += 0.5
        elif t in UNITS:
            current += UNITS[t]
        elif t in TENS:
            current += TENS[t]
        elif t in SCALES:
            if t == "hundred":
                current = (current or 1) * 100
            else:
                total += (current or 1) * SCALES[t]
                current = 0.0
        elif re.fullmatch(r"\d[\d,]*(?:\.\d+)?", t):
            current += float(t.replace(",", ""))
    value = total + current

    if tail:
        digits = "".join(str(UNITS[t]) for t in tail if t in UNITS and UNITS[t] < 10)
        if digits:
            value = float(f"{int(value)}.{digits}")
    return value


def numbers(text):
    """Canonical (value, is_percent) pairs, with magnitude suffixes applied.

    Returns a set for membership testing, so "$84K", "84,000", "84000" and
    "eighty four thousand" all collapse to 84000.0 and stop being reported as
    fabrications.
    """
    out = set()
    for _, digits, suffix in NUMBER_RE.findall(text):
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        suffix = (suffix or "").lower()
        if suffix in PCT_SUFFIXES:
            out.add(("pct", round(value, 4)))
        else:
            value *= MAGNITUDE.get(suffix, 1.0)
            if value:
                out.add(("num", float(f"{value:.6g}")))

    for m in SPOKEN_RE.finditer(text):
        raw = m.group(0).strip()
        is_pct = bool(re.search(r"(?:percent|%)\s*$", raw, re.I))
        core = re.sub(r"\s*(?:percent|%)\s*$", "", raw, flags=re.I)
        if not re.search(r"[a-z]", core, re.I):
            continue  # plain digits -- NUMBER_RE already has it
        value = _eval_run(core)
        if not value:
            continue
        if is_pct:
            out.add(("pct", round(value, 4)))
        else:
            out.add(("num", float(f"{value:.6g}")))
    return out


def _num_display(value, kind):
    return f"{value:g}%" if kind == "pct" else f"{value:,.0f}" if value >= 1000 else f"{value:g}"


def _name_candidates(generated):
    """Capitalized non-stopword tokens that are not sentence-initial."""
    out = []
    for sentence in SENTENCE_SPLIT.split(generated):
        for tok in sentence.split()[1:]:
            word = tok.strip(TRIM)
            if CAPITALIZED_RE.fullmatch(word) and word.lower() not in STOPWORDS:
                out.append(word)
    return out


def _norm_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def name_grounded(name, src_norm):
    """Is this proper noun recognisably in the transcript?

    Exact matching alone is wrong here, because the transcript is machine
    transcription: the summaries write "Clearinghouse", "EnergyLink", "Permian"
    where the ASR heard "clearing house", "energy line", "Perming". Comparing
    alphanumerics-only kills the spacing variants, and a five-character prefix
    catches the mishearings. Four-character names (e.g. "Weld") still need an
    exact hit, so an invented short county name is not let through.
    """
    n = _norm_name(name)
    if len(n) < 3:
        return True
    return n in src_norm or (len(n) >= 5 and n[:5] in src_norm)


def grounding(generated, source):
    """Count distinct figures and proper nouns in the output missing from the source.

    A proxy, not a fact-checker: it flags invented figures and invented names,
    which is where a small summarizer fails, and says nothing about claims built
    entirely out of real words. Both axes still over-report -- a figure stated in
    a form neither extractor covers, or a name the ASR mangled past a five-letter
    prefix, reads as invention. Treat it as an upper bound and hand-audit any
    number before quoting it.
    """
    src_norm = _norm_name(source)
    gen_nums = numbers(generated)
    bad_numbers = gen_nums - numbers(source)

    names = _name_candidates(generated)
    bad_names = [n for n in names if not name_grounded(n, src_norm)]

    examples = {_num_display(v, k) for k, v in bad_numbers} | set(bad_names)
    return {
        "ungrounded_numbers": len(bad_numbers),
        "ungrounded_names": len(bad_names),
        "n_numbers": len(gen_nums),
        "n_name_candidates": len(names),
        "examples": sorted(examples)[:8],
    }


def ends_mid_sentence(text):
    """Proxy for hitting a length cap: output stops without terminal punctuation.

    We have finished summaries, not token counts, so real truncation is not
    recoverable. This over-reports -- a summary may legitimately end on a colon.
    """
    t = text.rstrip()
    return bool(t) and t[-1] not in '.!?"\''


def bootstrap_ci(values, iters=BOOTSTRAP, seed=42):
    rng = random.Random(seed)
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0,) * 2
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def paired_delta(a, b, iters=BOOTSTRAP, seed=42):
    """Bootstrap CI on mean(a - b) over the same calls. Zero inside means tie."""
    diffs = [x - y for x, y in zip(a, b)]
    if len(diffs) < 2:
        return 0.0, (0.0, 0.0)
    lo, hi = bootstrap_ci(diffs, iters, seed)
    return statistics.fmean(diffs), (lo, hi)


def score_variant(rows, variant, scorer):
    out = []
    for row in rows:
        gen = row["variants"][variant]
        ref = row["reference_summary"]
        corpus = corpus_for(variant, row)
        s = scorer.score(ref, gen)
        out.append({
            "call_id": row["call_id"],
            "scores": {m: s[m].fmeasure for m in METRICS},
            "words": len(gen.split()),
            "reference_words": len(ref.split()),
            "duplicate_rate": duplicate_rate(gen),
            "copy_rate": copy_rate(gen, corpus),
            "empty": not gen.strip(),
            "mid_sentence": ends_mid_sentence(gen),
            **grounding(gen, corpus),
        })
    return out


def summarize(items, variant):
    mean = statistics.fmean
    stats = {"variant": variant, "n": len(items)}
    for m in METRICS:
        vals = [r["scores"][m] for r in items]
        lo, hi = bootstrap_ci(vals)
        stats[m] = {
            "mean": round(mean(vals), 4),
            "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
    stats["output_words"] = round(mean([r["words"] for r in items]), 1)
    stats["reference_words"] = round(mean([r["reference_words"] for r in items]), 1)
    stats["mean_duplicate_rate"] = round(mean([r["duplicate_rate"] for r in items]), 3)
    stats["mean_copy_rate"] = round(mean([r["copy_rate"] for r in items]), 3)
    stats["empty_outputs"] = sum(1 for r in items if r["empty"])
    stats["mid_sentence_ends"] = sum(1 for r in items if r["mid_sentence"])
    stats["ungrounded_numbers"] = sum(r["ungrounded_numbers"] for r in items)
    stats["ungrounded_names"] = sum(r["ungrounded_names"] for r in items)
    stats["numbers_emitted"] = sum(r["n_numbers"] for r in items)
    flagged = sum(1 for r in items if r["ungrounded_numbers"] or r["ungrounded_names"])
    stats["calls_with_ungrounded"] = flagged
    stats["pct_calls_with_ungrounded"] = round(100 * flagged / len(items), 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=os.path.join(HERE, "eval_set.jsonl"))
    ap.add_argument("--frozen", default=os.path.join(HERE, "EVAL_FROZEN.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "scores.json"))
    ap.add_argument("--stemmer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--corpus", nargs="*", default=[], metavar="VARIANT=FIELD",
                    help="override the grounding corpus for a variant, e.g. "
                         "student_plain=input_meta. Use when A/B arms read different "
                         "prompts and must be graded against the same factual record.")
    args = ap.parse_args()

    for spec in args.corpus:
        if "=" not in spec:
            sys.exit(f"--corpus entry must be variant=field, got {spec!r}")
        v, f = spec.split("=", 1)
        CORPUS_OVERRIDE[v] = f

    rows = [json.loads(line) for line in open(args.eval_set, encoding="utf-8") if line.strip()]
    if not rows:
        sys.exit(f"{args.eval_set} is empty")

    # refuse to score a run that does not cover the frozen set
    if os.path.exists(args.frozen):
        with open(args.frozen, encoding="utf-8") as f:
            frozen = json.load(f)
        got, want = {r["call_id"] for r in rows}, set(frozen["call_ids"])
        if got != want:
            sys.exit(f"eval set covers {len(got)}/{len(want)} frozen ids "
                     f"({len(want - got)} missing, {len(got - want)} extra) -- not comparable")

    variants = sorted({v for r in rows for v in r["variants"]})
    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=args.stemmer)

    per_variant, raw = {}, {}
    for v in variants:
        items = score_variant(rows, v, scorer)
        raw[v] = items
        per_variant[v] = summarize(items, v)

    paired = {}
    if len(variants) == 2:
        a, b = variants
        a_by = {r["call_id"]: r for r in raw[a]}
        b_by = {r["call_id"]: r for r in raw[b]}
        for m in METRICS:
            delta, (lo, hi) = paired_delta([a_by[c]["scores"][m] for c in a_by],
                                           [b_by[c]["scores"][m] for c in a_by])
            wins = sum(1 for c in a_by if a_by[c]["scores"][m] > b_by[c]["scores"][m])
            losses = sum(1 for c in a_by if a_by[c]["scores"][m] < b_by[c]["scores"][m])
            paired[m] = {
                "a": a, "b": b,
                "mean_delta": round(delta, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "crosses_zero": lo <= 0 <= hi,
                "wins": wins, "losses": losses, "ties": len(a_by) - wins - losses,
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_variant": per_variant, "paired": paired, "rows": raw},
                  f, indent=2, ensure_ascii=False)

    print(f"{'variant':<14}{'R1':>7}{'R2':>7}{'RL':>7}{'R1 95% CI':>18}"
          f"{'words':>7}{'dup':>6}{'copy':>6}{'mid':>5}{'ungrd#':>7}{'ungrdNm':>8}")
    for v, s in per_variant.items():
        ci = f"[{s['rouge1']['ci95'][0]:.3f},{s['rouge1']['ci95'][1]:.3f}]"
        print(f"{v:<14}{s['rouge1']['mean']:>7.3f}{s['rouge2']['mean']:>7.3f}"
              f"{s['rougeL']['mean']:>7.3f}{ci:>18}{s['output_words']:>7.1f}"
              f"{s['mean_duplicate_rate']:>6.2f}{s['mean_copy_rate']:>6.2f}"
              f"{s['mid_sentence_ends']:>5}{s['ungrounded_numbers']:>7}"
              f"{s['ungrounded_names']:>8}")
    if paired:
        print()
        for m, p in paired.items():
            verdict = "TIED (CI crosses 0)" if p["crosses_zero"] else "real difference"
            print(f"paired {m}: {p['a']} - {p['b']} = {p['mean_delta']:+.4f} "
                  f"CI [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]  "
                  f"{p['wins']}W/{p['losses']}L/{p['ties']}T  -> {verdict}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
