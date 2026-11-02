# Design — distilling the summarizer

Settled 2026-10-12, in a grilling session over the plan. Everything here is a
decision that was put to the user and answered, or a fact verified from the files
or from primary sources. Nothing here is assumed.

## Definition of done

A ~2B model that runs locally on a CPU-only Windows laptop and produces summaries a
reader would genuinely ship — **even if ROUGE drops**. Reported with two numbers
side by side: agreement with the teacher, and reference-ROUGE against the human
reference summaries.

## Decisions

**D1 · Teacher — the project's own Qwen3-8B LoRA.** `finetune/qwen3-8b-lora/`. It is
the voice being preserved, it is the model named in the mission, and it lives in
this repo. Its memorisation of 122 benchmark calls cannot reach the student, because
the student's training prompts are disjoint from the benchmark (D4).

**D2 · Student — Qwen3.5-2B.** Settled by measurement rather than preference: no
open-weight model in the 1.5–2B class is ≥1.5× faster at 4-bit. The best candidate
(`Qwen3-1.7B`) measures 1.23× at short context and is inferred *slower* at this
project's 3,620-token prompt, because Qwen3.5-2B's 18 linear-attention layers keep a
constant-size state while a plain transformer's KV cache grows. Every model that
does clear 1.5× is sub-1.5B and materially weaker on knowledge.

**D3 · Method — sequence-level distillation (Seq-KD).** Forced, not chosen. Teacher
vocabulary 151,669, student 248,320, independently learned BPE merge tables, so no
token-index correspondence exists and logit distillation is *undefined*. Text is the
only channel that crosses.

**D4 · Corpus — the 154-record holdout, already on disk.** 142 train / 12 validation,
at `call-summarizer/finetune/train.jsonl` and `val.jsonl`. Verified as a set
equality against `training-corpus − EVAL_FROZEN`. All 144 benchmark calls are
therefore clean, making the 2B result directly comparable to `qwen4b` (0.4410,
same split) and `qwen8b_honest` (0.4372).

**D5 · Targets — both, for every call.** 284 examples: 142 reference-targeted,
142 teacher-targeted. Doubles the data for a 2B, acts as a floor against the copy
bias that killed the 250M T5 attempt, and makes D6's control arm free.

**D6 · Two arms.** Blend (284 examples) against a reference-only control. **The
control must also see 284 examples** — the reference half duplicated once — so the
arms differ only in what the second half contains, not in data volume or step count.
Without that, a difference between the arms is uninterpretable.

**D7 · Teacher decoding — beam search, k=4, one pseudo-label per call.** Kim & Rush's
canonical "approximate" mode: cleanest targets, one inference pass, blend stays
exactly 50/50. Note this is a different job from `results.jsonl`, which used
sampling at temperature 0.7 to produce *deliverables*. If the student's copy rate
climbs, the named remedy is sampling several candidates per call and selecting for
informativeness *and* novelty (DisCal).

**D8 · Two metrics, teacher-agreement as the headline.** Reference-ROUGE (existing
`eval/score.py`) answers "still something a human would accept"; ROUGE(student,
teacher) on calls the student never trained on is the literal test of mimicry and
matches the stated definition of done. Requires a small new scorer — `score.py` is
frozen to the 144 and only emits paired CIs with exactly two variants.

**D9 · Student chat template — hand-rolled.** Thinking forced off inside the
template, and `{% generation %}` markers so `assistant_only_loss` has a span to key
off. Qwen3.5 dropped Qwen3's `/think` runtime switch, so this cannot be done at call
time.

**D10 · Student LoRA — mirrored from the 8B.** r=16, α=32, dropout 0.05,
`all-linear`, 3 epochs, same LR schedule. Keeps the arms comparable to each other
and to the 4B baseline.

**D11 · GPU work runs on Runpod**, price and shape quoted before anything is
created, pod deleted in the same session.

## Build order

Status as of 2026-10-12. Steps 1–8 are done; step 9 is the one item in flight.

1. Teacher inference pass over the 142 + 12 prompts. `make_infer_input.py` is
   hardcoded to the 144 deliverable calls via `cleaned_index.csv` and needs
   repointing. The training rows carry **no id field** — identity is recovered by
   content-joining each row's assistant target back to `dataset_raw.json`'s
   `call_id`. Fail loudly on a failed join rather than mislabel silently.
   *Done: 154/154 in 12.1 min, 4.7 s/call.*
2. Build the two datasets (blend 284, control 284). *Done.*
3. Two Runpod training runs. *Done: 108 steps each, 5.7 min on the pod.*
4. Write the teacher-agreement scorer. *Done — it lives in the sibling
   `call-summarizer/eval/` project, which is why its absolute levels cannot be
   re-derived from this repo even though its ordering can.*
5. Evaluate all 144 and compare against `qwen4b` 0.4410, `qwen8b_honest` 0.4372,
   T5-250M 0.212. *Done: the student scores 0.4399 ROUGE-1, and all four paired
   comparisons are in Lesson 04.*
6. Quantize to GGUF and measure on the actual laptop — prefill *and* decode.
   *Done: over all 144 deliverable calls, a median 16.0 s per call at 67% prefill
   (p90 26.6 s, max 33.0 s), at 163 tok/s prefill and 27.7 tok/s decode.*
7. Lessons 02, 03, 04. *Done.*
8. **Score the deployed Q4_K_M artifact on the same 144 calls.** Steps 5 and 6
   measured two different things — the 16-bit adapter's quality, and the
   quantized file's speed — so the artifact that actually ships had a measured
   latency and no measured quality. *Done: the deployed arm scores 0.428 ROUGE-1
   against the 16-bit arm's 0.440, a gap the paired bootstrap calls real on all
   three variants (+0.0114 [+0.0001, +0.0230] at ROUGE-1, 78W/66L). It is not the
   control arm's failure repeated — the deployed model's ungrounded names sit on
   26 of 144 calls rather than 443 of the control's 512 on one runaway loop — so it
   is a diffuse regression, not an outlier.*
   `verify_quant.py` could not supply the missing half: prefix agreement against a
   bf16 reference tracks decode *trajectory*, and one early argmax flip destroys it
   while leaving a perfectly good paraphrase. The path is `make_student_eval.py
   --samples q4km=local_samples.jsonl --only student_q4km student_blend`, then the
   sibling `eval/score.py`, which scores both arms against the same references
   under the same bootstrap and emits a paired interval.
   **Corrected 2026-10-13 — the grounding half of this step was wrong when it was
   published.** It read "324 ungrounded names on 139 of 144 calls at a median of 2".
   `student_q4km` was a *new variant name*, and step 7's `COPY_CORPUS` fix had been a
   name registration (`student_blend`, `student_control`), so the new name fell
   through to `cleaned_transcript` and re-inherited the exact bug step 7 had closed.
   Against the corpus the model actually read — `model_input`, byte-identical to the
   `infer_input.jsonl` user field on all 144 calls — the deployed arm writes **37**
   ungrounded names on **26** calls. ROUGE never depended on the corpus, so every
   quality number in this step is unchanged and the re-run reproduces the paired
   interval to the digit. The registry now refuses to guess (see step 7) and the
   deployed arm is registered.
9. **Separate quantization from the engine in that 0.012 gap.** Step 8 varies three
   things at once: the 16-bit arm ran inside transformers from an unseeded draw; the
   served run is llama.cpp through Ollama at seed 42; and only one of them is
   quantized. It does **not** vary sampling — both run temperature 0.7, top_p 0.9 and
   repetition penalty 1.05 (`student_generate.py:135-141`). An earlier draft of this
   file called the HF arm greedy, which that source contradicts. The control is
   therefore the f16 GGUF through the same engine at the same sampling and seed,
   which leaves precision as the only difference. It also replays the *same input
   file in the same order*, so the two Ollama runs line up call for call and each
   paired test is a within-call comparison rather than a mean over two sets. Two
   paired tests, not one: `f16` against `q4km` (both through Ollama — precision
   alone), and `f16` against `blend` (both 16-bit — engine and draw alone). Their
   sum is step 8's 0.0114.
   **Answered 2026-10-13.** The f16 GGUF ran all 144 calls through Ollama and the
   decomposition came out exact:

   | step | arms | delta ROUGE-1 | 95% CI | verdict |
   |---|---|---|---|---|
   | engine + draw | blend − f16 | −0.0048 | [−0.0164, +0.0067] | tied, 65W/79L |
   | precision alone | f16 − q4km | **+0.0162** | [+0.0057, +0.0266] | **real**, 83W/61L |
   | **sum** | blend − q4km | **+0.0114** | [+0.0001, +0.0230] | step 8, reproduced |

   So **the gap is the quantization.** Serving the same 16-bit weights through
   llama.cpp at a fixed seed costs nothing measurable — if anything it scores
   fractionally higher (0.4452 against 0.4399) while writing 16 words *fewer*
   (125.0 against 141.2). What is left, once the engine and the draw are held
   constant, is precision alone, and it is both larger and firmer than the number
   step 8 published. **The three-way comparison was diluting the effect:** the
   engine/draw term happens to pull the other way by 0.0048, inside its own noise,
   so the bundled figure read as +0.0114 with a lower bound at +0.0001 — "real by a
   whisker". Properly attributed it is +0.0162 with a lower bound of +0.0057. Nothing
   in step 8 was mis-measured; a number that answers a question it was not designed
   to answer is exactly what step 9 existed to unpack.
   Caveat worth keeping: seed 42 fixes the *draw*, not the output — the two models
   have different logits, so the token streams diverge. The seed removes the draw as
   a nuisance variable; it does not make the arms comparable token for token.
10. **Ask whether the invented names are a decoding artifact, by serving greedy.** This is
   a *new* question, not the control step 9 replaced: greedy is not what the HF arm does, so
   it cannot stand in for it. Run `measure_local.py --greedy` on the **shipped Q4_K_M** arm,
   against the sampled run already on disk. One variable changes — decoding alone, same
   weights, same engine, same seed — so the comparison is clean, and it is the arm that
   actually ships, so a win is a config change rather than a new build. What it settles:
   whether the ungrounded names that remain are sampling-induced (a one-line fix) or a
   property of the weights (a retraining problem), and whether the deployment can be made
   deterministic instead of seed-dependent. **The first of those got smaller on
   2026-10-13** — step 8's 324 became 37 once the grounding corpus was fixed, so the
   deployed arm now invents *fewer* names (37) than the 16-bit arm (46) and far fewer than
   its teacher (109). The question is still worth an hour, but it is now a question about a
   37-name residual rather than a 324-name one; determinism is the stronger reason to run
   it. Queued behind step 9: two Ollama runs at once contend for CPU and RAM, and a
   contended run's timings are not comparable to an uncontended one. (An earlier version of
   this line blamed that contention for a 33-minute stall in the f16 run. The event log says
   otherwise — it was a Modern Standby suspend. The queueing rule stands on its own, because
   it is about comparability, not about the stall.)
   `measure_local.py`'s `--greedy` docstring was corrected here — it had described greedy
   as "the way the HF/eval arms did", the same claim step 9 records as false.
   **Guard against a silent no-op:** both GGUFs carry `temperature 0.7 / top_p 0.9 /
   repeat_penalty 1.05` baked into their Modelfiles, so `--greedy` only works if Ollama
   prefers request options to Modelfile parameters. If it does not, the run reproduces the
   sampled run exactly and reads as *"decoding changes nothing"* — a wrong conclusion drawn
   from a broken switch. So sample three calls with `--limit 3 --greedy` into a scratch file
   and diff them against the matching sampled rows before committing the hour.
   **Guard passed 2026-10-13.** All three smoke calls differ from their sampled
   counterparts, and differ in a direction that rules out a trivial prompt-echo:
   greedy wrote 152 / 183 / 128 tokens where sampling wrote 91 / 157 / 97. Request
   `options` do override Modelfile `PARAMETER`s, so the switch is real and the hour
   was worth spending. The full run is on `student-blend` at `--keep-alive 0`, matching
   the shipped run's conditions exactly, so decoding is the only variable.
   **Full run completed and scored 2026-10-13.** All 144 calls, `--model student-blend`,
   `--keep-alive 0`, `--greedy`, into `finetune/local_samples_greedy.jsonl`. The tag was
   the right one: `ollama list` shows `student-blend:latest` at 1.3 GB, the Q4_K_M
   artifact, against `student-blend-f16:latest` at 3.8 GB. Throughput settled at a
   **median 18.5 s per call against the sampled run's 16.0 s** — 1.16× wall, of which
   1.07× prefill and 1.09× decode on the 143 calls neither run slept through. It is
   decoding slightly more (a median 143 output tokens against 132), which is the same
   direction as the three smoke calls and much smaller than they suggested; the smoke
   calls were a poor throughput predictor and are not used below.
   **The measured answer, `finetune/scores_greedy.json` against `scores_q4km.json`:**

   | metric | greedy | sampled | paired Δ | 95% CI | verdict |
   |---|---|---|---|---|---|
   | ROUGE-1 | 0.4323 | 0.4285 | +0.0038 | [−0.0101, +0.0170] | tied, 75W/69L |
   | ROUGE-2 | 0.1431 | 0.1312 | **+0.0119** | [+0.0035, +0.0193] | **real**, 91W/53L |
   | ROUGE-L | 0.2461 | 0.2369 | +0.0093 | [−0.0003, +0.0174] | tied, 91W/53L |
   | ungrounded names | 24 on 16 calls | 37 on 26 calls | — | — | no paired test |
   | ungrounded numbers | 76 | 105 | — | — | no paired test |
   | words (mean) | 108.7 | 108.0 | — | — | — |

   **The question this step was built to settle has a one-line answer: sampling-induced,
   and yes the deployment can be made deterministic.** Greedy cuts ungrounded names by
   about a third and ungrounded numbers by a quarter, and it does so at *identical
   length* (108.7 words against 108.0), which rules out the boring explanation that a
   shorter summary simply has fewer chances to invent. Nothing in the weights needed to
   change.

   **The discipline the numbers demand.** Only **one of the three** ROUGE variants is a
   real difference, so the defensible claim is *"greedy is not worse on ROUGE"* — no
   metric moved down — and the improvement claim has to rest on ROUGE-2 together with the
   grounding counts. It is worth saying plainly that "better on ROUGE" is **not** what
   this run shows and would be an overread of 1-of-3. The grounding counts carry **no
   paired test at all**: the 24-against-37 is a difference of totals, and it is not
   backed by a per-call interval the way the ROUGE figures are.

   **And it is a net reduction, not a strict improvement.** Call by call, sampling invents
   more names on **19** calls and greedy invents more on **8** (117 tie). Greedy's 24 names
   are not a subset of sampling's 37 — they are a partly different set of 24. So "greedy
   invents fewer names" is a statement about the distribution, not about any given call,
   and a reader who took it as "greedy is safer on your call" would be reading it wrong.

   **The honest cost, both halves.** One of the 144 greedy calls (`call_id 92093631892`)
   ran into the 900-token cap and ended mid-sentence — `done_reason: length`, the run's one
   `mid_sentence` row, where the sampled run had zero. Greedy has no escape from a loop
   it enters, so the failure mode it can have is the one sampling mostly avoids. Copy rate
   also rises slightly (0.0158 against 0.0117), the direction Seq-KD is documented to
   push. Against that: it is deterministic, which the sampled arm is not, and the
   seed-dependence noted in step 9 disappears as a deployment concern.

   **One row is a suspend, not a stall.** The run's wall time was 181.6 min rather than
   the ~75 min 144 × 18.5 s predicts, because call `112675989088` shows 8088.2 s wall of
   which **8080.5 s is prefill** against a 5.2 s decode. The Windows event log
   (`Microsoft-Windows-Kernel-Power`, IDs 506/507) shows **Modern Standby entered
   10:03:36 and exited 12:18:01 = 8065 s**, matching the prefill to within about 15 s; the
   call began just before the suspend. Zero prompts were served from cache, so nothing is
   a caching artifact. This is the second instance of the shape learning record 0003
   documents, and the standing pre-check in `NOTES.md` caught it *before* any explanation
   was written this time. The published figures are unaffected: the comparison above
   excludes that row (n=143), and the shipped sampled arm has no row over 40 s at all.
11. **Quantize to int8 and measure speed and accuracy.** Step 1 of the 2026-10-22 roadmap.
   `llama-quantize` produced a Q8_0 GGUF — 320/320 tensors, 3590 → 1908 MiB at 8.51 BPW —
   served through Ollama with byte-identical parameters to the shipped arm. **Score the
   accuracy answer first, because it is the good news and it is unambiguous:**

   | arm | ROUGE-1 | words | ungrounded names | ungrounded numbers |
   |---|---|---|---|---|
   | f16 | 0.4447 | 125.0 | 33 | 92 |
   | **Q8_0** | **0.4392** | 126.6 | **25** | **81** |
   | Q4_K_M (shipped) | 0.4285 | 108.0 | 37 | 105 |

   Paired bootstrap, Q8_0 against each neighbour:

   | comparison | ROUGE-1 | ROUGE-2 | ROUGE-L | reading |
   |---|---|---|---|---|
   | Q8_0 − Q4_K_M | +0.0104 [−0.0007, +0.0210] | **+0.0098 [+0.0031, +0.0162]** | **+0.0087 [+0.0007, +0.0165]** | 8-bit is better; R1 tied, R2/RL real |
   | f16 − Q8_0 | +0.0058 [−0.0038, +0.0157] | +0.0033 [−0.0031, +0.0098] | +0.0043 [−0.0036, +0.0119] | **all three tied** |

   **So 8-bit's quality cost is not detectable, and 4-bit's is.** That is the cleanest
   statement this project can make about quantization: the 0.0162 that step 9 attributed
   to precision is bought almost entirely by the *last* halving of the bit width, not
   spread evenly across it. If quality were the only criterion, Q8_0 would ship.

   **Speed is where it loses, and the result is counter-intuitive.** Over all 144 calls,
   token-weighted throughput on this CPU:

   | quant | size | prefill tok/s | decode tok/s | median wall |
   |---|---|---|---|---|
   | Q4_K_M | 1.27 GB | **162.8** | **27.7** | **16.0 s** |
   | Q8_0 | 1.92 GB | 82.2 | 19.3 | 32.4 s |
   | f16 | 3.78 GB | 101.7 | 13.9 | 29.0 s |

   **Q8_0 is slower at prefill than f16, which is nearly twice its size**, and it is 2.02×
   the shipped arm's wall time. A separate `llama-bench` run on the same machine
   reproduces the ordering independently (`bench_precision.log`: pp512 291 / 98 / 115
   tok/s, tg128 29.8 / 17.4 / 13.2 for Q4_K_M / Q8_0 / f16). The instrument gap is the
   usual third — different prompt lengths and kernel paths — but the ordering is stable,
   so this is a property of the backend, not of a contended run: **llama.cpp's Q8_0
   kernels are much worse on this architecture than either K-quants or f16.** Quantization
   is not a monotonic speed/quality dial; the operating points are set by which kernels
   exist, and the middle one here is the worst of the three.

   **Verdict: do not ship Q8_0.** It is the best of the three on grounding, tied with f16
   on ROUGE, and 2× slower than the shipped 4-bit for a quality gain no interval can
   resolve. The 4-bit artifact remains the right operating point, and the reason is now
   measured rather than assumed.
12. **Metadata A/B on the 2B: speaker name and deal info versus transcript-only.** Two
   LoRAs trained from the same base with the same recipe and seed on the same 284 rows,
   differing only in the user message (`build_meta_ab.py` asserts that every other field is
   identical, index for index). `plain` reads `Transcript:` only; `meta` reads
   `Prospect: <name>`, the deal stage and the deal outcome above it. Deal *size* is not a
   field in the corpus (`entities` is empty for 289 of 305 calls), so it was not tested.

   | comparison | ROUGE-1 | ROUGE-2 | ROUGE-L | ungrounded names | numbers | words |
   |---|---|---|---|---|---|---|
   | meta | 0.4435 | 0.1432 | 0.2421 | **39** | 95 | 138.9 |
   | plain | 0.4377 | 0.1388 | 0.2427 | **99** | 83 | 134.8 |
   | paired | tied | tied | tied | — | — | — |

   **ROUGE cannot see the difference; grounding can.** All three variants tie on the
   paired bootstrap (+0.0058 [−0.0041, +0.0157] at ROUGE-1, 76W/68L), while the metadata
   arm invents less than half as many names — 39 on 23 of 144 calls against 99 on 51, a
   *diffuse* reduction rather than one runaway call. Placed against the student that
   currently ships (`student-blend`, 0.4399), the metadata arm is again tied on all three
   ROUGE variants and slightly ahead on names (39 against 46), with numbers level (95/95).

   **One measurement trap this run exposed, now closed in the scorer.** The first pass
   grounded each arm against *its own* prompt, which reported plain at 117 ungrounded
   names — a 3× difference that was mostly an artifact: with the deal-stage line removed
   from plain's corpus, a summary that correctly *infers* a stage ("Marketing Agreement
   Request") is counted as invention. `score.py --corpus variant=field` now lets both
   arms be graded against the same fullest factual record; the figure above (99) is that
   corrected run. Any future A/B whose arms read different prompts must use it.

   **Verdict: metadata is worth having, but it is a grounding change, not a ROUGE change.**
   It halves invented proper nouns at no measurable cost in overlap with the reference.
   The next step is the one the roadmap names: distill the winning configuration into 2B
   properly, then move to the next target output.
13. **Deploy the metadata model and measure it against the shipped artifact.** Merge →
   GGUF → Q4_K_M (1.27 GB, the same footprint) → Ollama with identical parameters, then
   the same 144 calls on the same laptop. The deployed comparison is the one that matters,
   because both arms are now served the same way:

   | arm | ROUGE-1 | ROUGE-2 | ROUGE-L | words | names | numbers | median wall |
   |---|---|---|---|---|---|---|---|
   | **student-meta** (Q4_K_M) | **0.4406** | **0.1422** | **0.2457** | 111.6 | **32** | **82** | 20.2 s |
   | student-blend (shipped) | 0.4285 | 0.1312 | 0.2369 | 108.0 | 37 | 105 | 16.0 s |
   | paired Δ | **+0.0121 [+0.0020, +0.0221]** | **+0.0110 [+0.0040, +0.0178]** | **+0.0088 [+0.0020, +0.0155]** | — | — | — | 1.26× |

   **This is the first change in the project to beat the shipped artifact on all three
   ROUGE variants at a real paired difference** (88W/56L, 85W/59L, 84W/59L), and it does so
   while inventing fewer names and markedly fewer numbers. It is also a far better trade
   than the f16 fallback: **1.26× the latency for +0.0121**, against f16's 1.85× for
   +0.0162. At 0.4406 the deployed 4-bit model now sits within 0.0041 of the 16-bit arm
   (0.4447) at two-thirds of its speed cost.

   Two limits on the claim. First, the arms do not receive the same input — the metadata
   model is *given* the prospect name, stage and outcome, so part of the gain is better
   conditioning rather than a better model, and that metadata must be available at
   serving time (it is: it comes from the CRM row already in Supabase). Second, step 12
   showed meta ≈ plain on ROUGE in transformers, so the deployed +0.0121 is against the
   *shipped* arm rather than the transcript-only control; the honest summary is "metadata
   plus this training run beats the shipped blend", not "metadata alone explains 0.0121".

   **The shipped artifact is now the weaker of the two.** Whether to switch is a product
   decision (26% slower per call, richer input requirement) rather than a measurement
   question, which is the point at which the project should stop and hand over.
14. **The first non-summary target: Next Steps.** The roadmap's "rinse and repeat" starts
   here, and the task has no purpose-built label column, so the labels were *extracted*
   from text the project already had: the "Next step(s):" sentence inside
   `dataset_raw.json`'s `target_summary` (human), falling back to the same sentence in
   `teacher_output.jsonl` (pseudo-label) only where the human reference has none. That
   yields **116 training calls** (82 human, 34 teacher-pseudo) and **75 benchmark calls**
   with a human reference — the other 69 have no next-step sentence at all, so this task
   is scored on 75 calls, not 144. A 2B LoRA trained in 3.2 min on 39 steps.

   | arm | ROUGE-1 | ROUGE-2 | ROUGE-L | words | names | numbers |
   |---|---|---|---|---|---|---|
   | **nextsteps LoRA** | 0.302 | 0.069 | **0.208** | **32.6** | 4 | **0** |
   | shipped summarizer, zero-shot | **0.316** | **0.075** | 0.181 | 65.8 | 7 | 20 |
   | paired Δ (LoRA − zero-shot) | −0.0139 [−0.0374, +0.0091] | −0.0063 [−0.0206, +0.0086] | **+0.0270 [+0.0090, +0.0460]** | — | — | — |

   **The dedicated model wins on structure and concision, not on word overlap.** ROUGE-1
   and ROUGE-2 tie (both point estimates slightly against it); ROUGE-L is a real gain at
   47W/28L; and it writes **half the length** with **zero ungrounded numbers** against the
   baseline's 20. That is the shape you want for a CRM "Next Steps" field — a sentence,
   not a briefing — and the zero-shot baseline's higher unigram overlap is partly the
   familiar artifact of a longer text happening to hit more reference words.

   **Two defects this pilot exposed, both fixed before the numbers above.**
   First, the label extractor ate the sentence terminator at the section cut, so all 75
   outputs of the first run ended without punctuation (`mid_sentence` flagged 75/75 — a
   formatting artifact, not truncation) and the target style was wrong. The cut now keeps
   the terminator and 75/75 end with a period. Second, the first baseline pass died with
   an Ollama HTTP 500 on call 74 of 75; the two missing calls were re-run rather than
   dropping them, because a partial join silently shifts the mean.

   **Caveats that limit the claim.** The training calls are much shorter than the eval
   calls (median 251 words against 1,034), a pre-existing property of this corpus that the
   summarizer also faced, so this is as much a generalisation test as a task test. And 116
   training rows is a third of what the summarizer used; label expansion (a teacher pass
   over the calls with no next-step sentence) is the obvious next lever if this task
   graduates from pilot.
15. **Test the 6-bit recommendation from a public GGUF card (2026-11-02).** A popular
   quantisation card (`mradermacher/Mixture-Summarizer-Qwen3.5-2B-GGUF`) labels **Q6_K**
   "very good quality" and recommends Q4_K_M as "fast, recommended". The suggestion worth
   testing was Q6_K: it is a *K-quant*, the family that was fast here, unlike Q8_0. Built
   from `student-meta-f16.gguf` and measured on the same 144 calls:

   | arm | size | ROUGE-1 | ROUGE-2 | ROUGE-L | words | names | numbers | median wall |
   |---|---|---|---|---|---|---|---|---|
   | Q4_K_M (shipped) | 1.27 GB | 0.4406 | 0.1422 | 0.2457 | 111.6 | 32 | 82 | **20.2 s** |
   | Q6_K | 1.56 GB | 0.4415 | 0.1401 | 0.2445 | 131.4 | 24 | 89 | 33.4 s |
   | paired Δ | — | −0.0009 [−0.0114, +0.0095] | +0.0021 [−0.0047, +0.0088] | +0.0011 [−0.0070, +0.0088] | — | — | — | **1.65×** |

   **All three ROUGE variants tie; the cost is 1.65× the wall time and 23% more disk.**
   The only movement is in grounding — 8 fewer invented names, 7 more invented numbers —
   and that does not survive as a claim without a paired test. `llama-bench` on the same
   machine agrees that Q6_K's prefill path is weak here (pp512 193.6 tok/s against
   Q4_K_M's 255.9, and **Q5_K_M's 272.1**), so this is the same kernel effect that made
   Q8_0 lose, not an artefact of one run.

   **Verdict: Q4_K_M remains the shipping point.** The public card's advice is generic and
   is about *quality-per-bit*, which is true here too — Q6_K is not worse — but the
   decision on this hardware is quality-per-*second*, and a tie at 1.65× the latency is not
   a trade worth taking. **Q5_K_M is the only quant on the bench that beats Q4_K_M's
   prefill while costing 8% on decode**, so if a higher-quality point is ever wanted, that
   is the one to test next — not Q6_K.




## Open risks

- **Copy rate.** Seq-KD is documented to reduce abstractiveness. Testable here
  because `score.py` already reports copy rate. Remedy named in D7. **Evidence since
  (2026-10-13):** the decoding control moved it the other way — greedy scores 0.0158
  against sampling's 0.0117 on the shipped artifact — so the configuration that ships
  is already the more abstractive of the two, and the risk is the one step 10 did not
  *introduce*. See `finetune/scores_greedy.json`.
- **Prefill dominates CPU latency — measured now, not projected.** Across all 144 deliverable
  calls on the deploy target, a median call costs 16.0 s with 67% of it prefill (p90 26.6 s,
  max 33.0 s), at 163 tok/s prefill and 27.7 tok/s decode. Prefill is still the larger half
  and it scales with the full prompt length, so caching helps only exact repeats. The residual
  question is no longer "is it fast enough to run" but "is 16 s per call acceptable at 291
  calls" — about 78 minutes of CPU for the full set, run strictly one at a time. **There is
  now a latency knob, and it costs (2026-10-13):** the greedy configuration that cuts
  invented names by a third runs at a median 18.5 s, so choosing it makes the full set
  roughly 90 minutes rather than 78. The two levers — precision and decoding — are the only
  ones this project measured a cost for, and both trade quality-of-output for time.
- **7 of the 144** benchmark calls were never in any training corpus at all (one
  dropped by the missing-field gate, six by the outlier filter), so even the
  contaminated 8B had not seen them.
- `EVAL_FROZEN.json`'s `note` field claims all 144 were in the 8B training rows. It
  is false — 137 were. Left unedited because it is a frozen artifact.

## Revisions during the build (2026-10-12)

Recorded separately rather than folded into the decisions above, because measurement
changed these and the reasoning is the useful part.

**D9 is superseded — there is no hand-rolled student template.** Building it showed why
the idea was unsafe. The first transcription of the shipped template spelled the empty
thinking block as the literal word `think` preceded by a space; that tokenizes to 7047.
The model's own template uses the 7- and 8-character angle-bracketed markers, which
tokenize to the added tokens **248068 / 248069**. The two renders agreed on 44 of their
45 tokens — the single difference sits at the first position of the assistant turn and
is invisible in a printed string. That is the same defect class as the Qwen3-8B
prompt-shape mismatch (train.py's template, token 7274, against predict.py's, 151667).

`train_student.py` therefore holds no template constant at all. It renders through the
model's own `chat_template` and derives the supervised span by re-rendering the prompt
half, asserting `prompt == full[:len(prompt)]` on every row. No `{% generation %}`
markers, no `assistant_only_loss` flag. `verify_student.py`'s old constant is kept only
as the fixture that caught the error and is marked superseded.

**D10 amended — LoRA targets are the text-side linear leaves, not `all-linear`.**
Qwen3.5-2B is not a text model: it is `Qwen3_5ForConditionalGeneration`, with a vision
tower (depth 24, hidden 1024, patch 16), `image_token_id` 248056 and `video_token_id`
248057. `all-linear` would hang adapters on the vision projections (`linear_fc1`,
`linear_fc2`, `proj`, `qkv`) that text-only data never gives a gradient to. `lm_head` is
excluded as well, because the student ties word embeddings. LoRA attaches to the 12
text-side leaves instead. Everything else in D10 stands.

**D2's tokenizer figure, corrected.** The student's vocabulary has three different
numbers depending on which one you ask: `len(tokenizer)` = 248,077, `vocab_size` =
248,044, `config.vocab_size` = 248,320. D3's argument is unaffected — the point is that
it does not correspond to the teacher's 151,669 — but the 248,320 quoted there is the
config value, not the tokenizer's.

**Latency risk, corrected.** The `~112 s prefill / ~21 s decode` figure above is the
3,620-token *maximum* prompt quoted as if typical. Across the 144 deliverable calls the
median call is ~51.5 s prefill and ~22.0 s decode (~70% prefill, ~74 s per call), and
the 90th percentile is ~95.4 s / ~31.2 s (~75% prefill, ~127 s per call). Prompt length
is still a first-class lever; the earlier figure overstated the median by about 1.5×.
**(Superseded below.** The RK3588 numbers above were never a measurement of this laptop at
all. The deploy target measures a median 16.0 s per call at 67% prefill, p90 26.6 s, at
163 tok/s prefill and 27.7 tok/s decode — see "The RK3588 latency projection is
retired".**)**

**The teacher pass is done: 154/154 in 12.1 min, 4.7 s/call.** Median 146 tokens, no
empties, no duplicate ids, exact call_id agreement with `teacher_input.jsonl` in both
directions. Two rows (`112780610566`, `116979700381`) hit the 900-token cap and were
degenerate rather than merely long. Both were in the **train** split, so both would have
become blend targets, and a degenerate target teaches the student to repeat itself. They
were regenerated with `--repetition-penalty 1.1` (209 and 78 tokens, terminating
naturally) and spliced in; the pre-splice file is kept as
`teacher_output.before_redo.jsonl`. This is a decoding-condition change on 2 of 154 rows,
accepted because a degenerate label is far worse than a slightly uneven one.

*Corrected later the same day.* This paragraph first read "12.6 min, 4.9 s/call" and
described the second row as having "emitted its entire summary twice". Both were wrong.
The live file measures 725 s (12.1 min, 4.7 s/call), and the second row's stored text is
364 characters — **byte-identical before and after the repair** — so it never contained a
duplicated summary. What it actually did is the more interesting finding, and it is
recorded at the end of this section.

**D6's control arm, corrected in construction.** The first build reached 284 rows by
concatenating the reference half twice (`[refs, refs]`), while the blend interleaves
(`[ref, teacher, ref, teacher, …]`). Both have 284 rows and the same step count, but the
control put every reference at a different *index* than the blend did, so the arms
differed in more than the content of their teacher rows — the exact confound D6 exists to
prevent. The control is now duplicated **in place** (`[ref₀, ref₀, ref₁, ref₁, …]`),
making index *i* a reference in both arms and leaving only the odd indices different.
Verified: same call_id, same prompt, and identical messages at every even index.

**Two engineering traps on the student, both from assuming a text-only model.**
`apply_chat_template(tokenize=True)` returns a `BatchEncoding`, which is **not** a `dict`
subclass — so an `isinstance(…, dict)` test fell through and iterating it yielded its
*keys* (`input_ids`, `attention_mask`) instead of tokens. Both renders collapsed to the
same two strings, the prefix assertion passed vacuously, and the guard surfaced it as
"the assistant turn contributed no tokens" — a corruption caught by a check designed for
something else. Separately, `Qwen3_5Config` has no top-level `pad_token_id` or
`eos_token_id`; those live on `text_config`, and reading the top-level name raises
`AttributeError` rather than returning `None`. Both are fixed, and both are the same
lesson as D9: verify the shape you actually got, never the shape you expected.

**The two arms trained, and the control overfits harder than the blend.** Both ran 5.7 min
— 284 rows, 108 optimizer steps, identical hyperparameters, seed and schedule. Validation
loss by epoch:

| | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| blend | 1.5788 | 1.5966 | 1.7093 |
| control | 1.6328 | 1.9158 | 2.3260 |

Both rise from epoch 1, so 3 epochs is more than 284 rows wants; the blend's rise is mild
(+0.13 nats) and the control's is steep (+0.69). That gap is a consequence of D6's own
construction rather than of label quality: matching data *volume* by duplicating the
references means the control's second half repeats its first, so each epoch is two passes
over 142 unique targets, while the blend's is one pass over 284 genuinely distinct ones.
The control therefore sees half the unique content at the same step count and has more to
memorise per unique row. Read the control's steeper curve as repetition, not as evidence
that human references are worse supervision — the arm is still the right equal-volume
control, but this is the one axis on which the two arms are not matched, and it is
unavoidable.

Val loss over 12 rows is also a weak proxy for the thing the mission actually asks for.
The three checkpoints a 3-epoch run pass through are not saved — only the final adapter is
— so if the evaluation shows the student underperforming the baselines, the named remedy is
a 2-epoch retrain measured against this one, not a guess from this table.

**Step 5 needs no new scorer.** `call-summarizer/eval/score.py` reads an `eval_set.jsonl`
of `{call_id, variants: {…}, reference_summary}` rows, and already carries `bart`,
`qwen4b`, `qwen8b_honest` and `qwen8b_lora`. Adding the student as a fifth variant and
re-running the frozen scorer puts it in the same table from the same references under the
same bootstrap — which is the only way its number is comparable to 0.4410 and 0.4372. The
scorer also already reports `copy_rate` (the DisCal failure mode D7 names) and
`mid_sentence_ends`, so the two risks this design is most exposed to are measured by the
existing harness rather than by anything new. It emits its paired bootstrap only with
exactly two variants, so `make_student_eval.py --only A B` writes a two-variant subset for
the significance test.

**A new variant name silently changes what a metric means.** `score.py` keeps a
`COPY_CORPUS` registry mapping each variant to *the text it actually read*, and
`score_variant` looks it up with `COPY_CORPUS.get(variant, "cleaned_transcript")`. That
default is not the right answer for a new Qwen variant: every existing Qwen entry maps to
`model_input`, which is what the student was prompted with too — the same
`infer_input.jsonl` `{system,user}` pair `predict.py` renders, and `model_input == user` on
all 144 calls, so the student saw byte-identical prompt content to `qwen4b` and
`qwen8b_honest`. Registering `student_blend`/`student_control` under `model_input` is
additive: the four baseline rows reproduce unchanged, confirmed by re-running the frozen
scorer and diffing.

The size of the error is why this is recorded. Against the wrong corpus the student's
`ungrounded_names` was **413**; against the right one, **46**. `cleaned_transcript` is raw
ASR with anonymised `SPEAKER_00`/`SPEAKER_01` labels, so the *call participants' own names*
never appear in it, and every summary that names the person on the call — which is most of
them, and what the house reference does too — reads as invented. `model_input` carries the
diarised names (`Alex Morgan: …`) and a trailing `(The prospect on this call is Jerry
Kennedy.)`, so grounding works as intended there. A 9× swing on a fabrication metric,
produced by nothing but a lookup default, on the one metric the design leans on to catch
the failure ROUGE cannot see. Hand-audit the grounding number before quoting it, which is
what its own docstring already says.

**2026-10-13 — the note above did not prevent the same bug, because the fix was a name.**
Registering `student_blend`/`student_control` fixed those two variants and nothing else. Step
8 then scored the deployed artifact as `student_q4km`, a name that did not exist when this note
was written, and it fell through to the same default — publishing 324 where the matched corpus
gives 37. The note was correct, was read, and made no difference, because "the default is wrong
for a new name" is a defect in the lookup, not in any particular name. `COPY_CORPUS.get(...)`
is now `corpus_for(...)`, which stops with the variant name instead of choosing a corpus, so
the failure mode is a failed run rather than a plausible number. Verified additive: the frozen
four-variant `scores.json` regenerates byte-identical.

**Step 5's first result: the student matches a 4B, on 2B.** `student_blend`, 144 calls,
scored by the frozen harness off the same references and the same bootstrap:

| variant | R1 | R2 | RL | copy | mid | ungr# | ungrNm |
|---|---|---|---|---|---|---|---|
| bart | 0.201 | 0.045 | 0.125 | 0.14 | 47 | 15 | 1 |
| qwen4b | 0.441 | 0.142 | 0.248 | 0.01 | 1 | 98 | 47 |
| qwen8b_honest | 0.437 | 0.142 | 0.242 | 0.01 | 6 | 101 | 107 |
| qwen8b_lora (teacher) | 0.478 | 0.182 | 0.285 | 0.01 | 0 | 92 | 109 |
| **student_blend** | **0.440** | **0.143** | **0.246** | 0.01 | 0 | 95 | **46** |
| student_control | 0.418 | 0.126 | 0.229 | 0.01 | 1 | 93 | 512 |

**Distillation beat the control, and the student tied the 4B.** Both claims are paired
bootstrap tests on the same 144 calls, not comparisons of overlapping intervals:

    student_blend - student_control   R1 +0.0223 CI [+0.0082,+0.0368]   84W/60L/0T  real
                                      R2 +0.0174 CI [+0.0097,+0.0251]   86W/58L/0T  real
                                      RL +0.0168 CI [+0.0076,+0.0263]   86W/58L/0T  real
    qwen4b - student_blend            R1 +0.0011 CI [-0.0114,+0.0131]   81W/63L/0T  TIED
                                      R2 -0.0005 CI [-0.0084,+0.0073]   73W/71L/0T  TIED
                                      RL +0.0022 CI [-0.0069,+0.0112]   81W/63L/0T  TIED

So training on the teacher's summaries is worth +0.022 R1 over training on the human
references at matched volume — the experiment's central claim, and it holds with the win
split 84/60 rather than a few outliers carrying it. And a 2B student is *statistically
indistinguishable* from the 4B baseline: every CI crosses zero, with win/loss near even.
D8's mimicry metric agrees in direction — ROUGE(student, teacher) is 0.4892 for the blend
against 0.4606 for the control, 0.4985 for `qwen4b` and 0.5008 for `qwen8b_honest`.

Two of the design's named risks did not materialise. Copy rate is 0.01, identical to the
baselines, so Seq-KD's documented abstractiveness loss (D7's DisCal worry) is not visible
here. And no blend output was empty, ended mid-sentence, or hit the 900-token cap on 144
calls.

**The control's 512 ungrounded names are one output, not a property of the arm.** Call
`116308713889` alone contributes 443 of them — it is a repetition loop (`Royalty Royalty
Royalty …`), which is also the arm's one mid-sentence end and its one cap hit. Excluding
it the control is 69 (4.0% of name occurrences) against the blend's 46 (3.0%) and the
teacher's 109 (6.4%). It also cost the control only 0.0026 R1, so the +0.022 distillation
effect is not an artifact of one bad call. Worth stating because the raw aggregate is
misleading in a way that would have flattered the blend for the wrong reason. That the
control is the arm that produced a repetition loop at all is consistent with its training
set: D6's volume-matching halves its unique content, and repetition is what a model does
when it has more to memorise per unique row.

**D10's LoRA attachment closes arithmetically, which is how "vision frozen" stops being an
assertion.** `r = 16`, so each adapted linear contributes `16 × (in + out)`. With the shapes
the config implies — `hidden 2048`, `intermediate 6144`, `full_attention_interval 4` putting
full attention at layers 3, 7, 11, 15, 19, 23:

    6 full-attention layers     : q 16x6144 + k 16x2560 + v 16x2560 + o 16x4096  =   245,760 each
    18 linear-attention layers  : qkv 16x8192 + z 16x4096 + a 16x2064 + b 16x2064
                                  + out 16x4096                                   =   328,192 each
    24 layers of MLP            : gate + up + down, 3 x 16x8192                   =   393,216 each

    6 x 245,760  +  18 x 328,192  +  24 x 393,216  =  16,819,200

That is exactly the trainable count the run reported. So the suffix match selected the 12
text-side leaves in precisely the layers expected and nothing else: had the vision tower's
`out_proj` been caught by the same suffix, or had `q_proj` matched a linear layer, the total
would not close. Two config details are load-bearing for later steps. `attn_output_gate: true`
is why `q_proj` is `[4096, 2048]` and not `[2048, 2048]` — it emits query and gate together.
And `mamba_ssm_dtype: "float32"` is the checkpoint declaring for itself the same thing the
quantizer's 2-D rule preserves: the SSM state tensors want F32, which is why refusing to
quantize `A_log [16]` and `dt_bias [16]` is not merely conservative but correct.

**Step 6: the base model was never corrupt — the published hash was the wrong hash.**
The 4.55 GB shard landed at exactly the right length and hashed to `aa33250c…` against a
"published" `4a33a6bd…`. Three cheap checks were run before spending another download on
it, and all three agreed with corruption: the Hub API and a HEAD on both `main` and the
pinned revision returned the same ETag, a local zero scan over 4.55 GB found **0 hole
regions**, and 21 evenly spaced 64 KB probes against the CDN **all agreed**. Every one of
those answers is equally consistent with the file being *fine*. They were run to confirm a
hypothesis rather than to distinguish it from the alternative, and not one of them could.

*Corrected after the re-download.* The test that did distinguish them was the one nobody
had run: ask the Hub which hash it publishes as the *content* hash. `GET
/api/models/Qwen/Qwen3.5-2B/tree/<rev>` returns **two** hashes for the same file —
`lfs.oid` = `aa33250c…`, the sha256 of the bytes, and `xetHash` = `4a33a6bd…`, Hugging
Face's Xet content hash, which lives in a different hash space. `EXPECT_SHA` had been
transcribed from the wrong field. The ETag derives from the Xet path, which is why it
appeared to disagree as well. The re-download settled it independently: 543 blocks,
**0 differing**, and the file's own sha256 equals `lfs.oid`. Structural validation agrees
— 632 tensors matching the index's 632, the highest tensor end offset equal to the file
size, the sum of tensor bytes equal to the index's `total_size`, and torch reading real
tensors across the range including the `mtp.` block. Nothing was ever damaged: a
10.5-minute re-download and a corruption hunt were spent on a wrong constant.

What survives is the rail, and it is worth keeping for any large resumed fetch. The first
downloader's real defect was that it set a `Range` header and trusted the answer. It never
checked `status == 206`, never checked that `Content-Range` began at the offset it asked
for, and never checked a block arrived complete. A CDN that answered a ranged request with
a 200 — or from a different start — would have had its body appended at the wrong position:
silent corruption. `refetch_verified.py` re-fetches in 8 MB blocks, requires 206 *and* a
matching `Content-Range` *and* an exact length on every one, and compares each block
against the old file as it writes, so a single pass returns both a verified file and a map
of any damage. It caught a `ConnectionResetError` at 948 MB and retried that block; the
first downloader would have appended whatever arrived.

**The quantizer's Qwen3.5 exposure is `in_proj_a`/`in_proj_b`.** Qwen3.5-2B is a hybrid:
24 text layers, of which 18 are Gated DeltaNet (`in_proj_a`, `in_proj_b`, `A_log`,
`dt_bias`, `conv1d`) and 6 are full attention, plus a 1-layer MTP head — 632 tensors, 25
blocks after conversion. Reading `tensor_allows_quantization` shows the guard that keeps
small SSM state tensors in F32 refuses anything under 2-D, so `A_log [16]` and
`dt_bias [16]` cannot be quantized, and `_norm.weight` and `ssm_conv1d` are excluded by
name. But the `ssm_f_a`/`ssm_g_a`/`ssm_beta` exclusions sit inside
`if (arch == LLM_ARCH_GLM5_NEXT)` and do **not** cover QWEN35. The live exposure is
therefore the two 2-D `[16, 2048]` projection matrices: 2048 is eight Q4_K superblocks, so
they quantize, and they feed exponential decay gates where relative error amplifies.
`verify_quant.py` turns that from an argument into a measurement — greedy-decode from the
merged bf16 checkpoint and from the deployed GGUF, then compare decoded *text* by longest
common character prefix. Text rather than token ids because the Ollama path is
detokenize → retokenize, and a byte-level merge difference would read as a false
divergence at a token index while the characters are identical.

**The verdict rule this paragraph originally ended with is superseded.** It read: *"A model
whose weights survived diverges late or in the tail; one whose state tensors were wrecked
diverges in the first few characters."* The second half is fine; the first half is false
for this stack, for reasons stage 2 had already measured, and a threshold built on it
fails a healthy model. The instrument now reports three distances and judges only against a
measured quantization-free control. See the quant-parity entry at the end of this section.

**QWEN35 conversion handles the MTP head, but only because of the top-level
architecture string.** The checkpoint carries `mtp.layers.0.*`, and `no_mtp` is set only
when the text architecture resolves to exactly `Qwen3_5TextModel`. Ours is
`Qwen3_5ForConditionalGeneration` with `text_config.architectures = None`, so `no_mtp`
stays False, `_QwenMtpMixin.filter_tensors` remaps `mtp.layers.N.X` →
`model.layers.{24+N}.X`, and `mtp_num_hidden_layers = 1` raises `block_count` to 25 — the
MTP head becomes `blk.24.*` rather than killing the converter at `Can not map tensor`.
Had the config named the text model at the top level instead, the same weights would
convert to a different graph.

**The template risk is closed by not using Ollama's template engine at all.**
`make_ollama.sh` writes no `TEMPLATE` directive deliberately — Ollama's directive is Go
template syntax and would not accept the Jinja the student trained with, and
hand-translating it is the exact mistake class that has already cost this project once.
Instead `measure_local.py` sends `raw: true` with the prompt rendered by the student's own
HF tokenizer, so the bytes reaching the model are the bytes `train_student.py` supervised
on. `--check-template` guards the seam by comparing Ollama's `prompt_eval_count` against
the HF tokenizer's count for the same render.

**A local latency baseline, measured on this laptop before the student exists.** The only
Qwen3.5 GGUF already on the machine (0.8B, Q4) gives ~38 tok/s decode and ~5 s prefill on
~1,700-token prompts (~340 tok/s prefill), a median call of 9.4 s with 53% of it spent in
prefill. Prompt length is confirmed as a first-class lever on x86 as it was on the RK3588.
Scaling by parameters puts a 2B Q4_K_M near 15 tok/s decode, so a ~900-token summary is
roughly a minute per call — which is why step 6 measures prefill and decode separately
rather than reporting one wall-clock number. **(Superseded: the 2B is measured at 163 tok/s
prefill and 27.7 tok/s decode, and its own summaries run ~105–157 tokens rather than ~900, so
an actual call costs a median 16.0 s at 67% prefill. The scaling prediction was the wrong
shape — it assumed the `num_predict` cap was the typical output length, and it was also wrong
about decode speed by nearly 2×, in the pessimistic direction.)**

**The second capped label was a loop you could not read, which is why the cap is the
audit.** `audit_teacher.py` flags two symptoms independently — a token count sitting on
the cap, and a repeated n-gram — because they catch different failures. Row
`112780610566` is the visible shape: 900 tokens, 3,951 characters, sliding into
`Lynn (agreement contact),` 106 times, and the n-gram test finds it. Row
`116979700381` is 900 tokens that produced only **364 characters** (70 words) — a
perfectly ordinary summary whose worst 6-gram occurs exactly **once**, so every textual
test passes it. Two independent measurements say the 900 tokens are real rather than a
recording bug. `23.75 s` is the unique maximum of the `seconds` column across all 154 rows
and exactly these two rows hold it; and 900 ÷ 23.75 = 37.9 tok/s for both, which sits just
above the healthy rows' p90 of 37.1 tok/s, where the rate lands once per-call overhead is
amortised over maximum work. A mis-recorded count cannot make a wall clock run long. So
the second row burned the full cap on tokens that produced no visible text, and
regenerating it returned the same 364 characters — nothing about its *text* was ever
wrong.

The consequence is the design point, and it is why the audit exists as a script rather
than a reading pass: a text scan for repetition catches one of these two rows and is
**structurally blind** to the other, so pseudo-labels cannot be audited by inspecting
them. The cap is not a redundant second signal; for this class of failure it is the only
detector there is. The repaired pass audits 0 flagged of 154 with max 275. The two rows
cost 47.5 s of the pass's 725 s — 6.5% of the wall clock for 1.3% of the rows.

**Step 6, stage 1: the merge makes two silent changes, and one of them is provably
lossless.** Both were found by reading the artifact rather than trusting
`merge_and_unload()`, and both were fixed there.

*The dtype policy.* The base stores 36 tensors as F32 — 18 `linear_attn.A_log`, 18
`linear_attn.norm.weight` — and its own config declares `mamba_ssm_dtype: "float32"`.
Loading it with `dtype=torch.bfloat16` rounds them, and `save_pretrained` then writes the
whole model in one dtype, so the declaration stops being true of the file. Measured
before fixing: `A_log` is **exactly** bf16-representable here (0.000e+00 error, in `A_log`
*and* in `exp(A_log)` — which matters, because `A_log` feeds an exponential decay per
recurrent step, so an error there would compound), while `norm.weight` loses up to
**3.891e-03** relative. Neither tensor is a LoRA target, so the adapter never touched
them and the correct merged value *is* the base value. That is what makes the restore
exact rather than approximate, and why its rail is an equality assertion rather than a
tolerance: if a future adapter ever does target one of these, the merge raises instead of
silently discarding adapter output. `merge_lora.py` copies the base's exact F32 values
back (not by loading the base in float32, which would cost 9.1 GB *and* change the merge
arithmetic for every other tensor), then re-reads the artifact's safetensors header to
confirm the F32 set survived the save. The run reports `restored 36 tensors to the base's
F32 (dtype policy preserved)` and `artifact carries 617 tensors, 36 of them F32`.

*The MTP head.* `AutoModelForImageTextToText` does not own `mtp.*`, so all 15
tensors (~121.7 MB) are dropped at load and never written. Plain autoregressive decoding
does not use them. The arithmetic closes both ways: **617 written + 15 dropped = 632**,
the base's tensor count, and 4.43 GB against the base's 4.548 GB is a 118 MB gap that is
the MTP block. It is now reported rather than left to be inferred from a file size.

**Step 6, stage 2: the old check could not have passed, and that is a property of bf16,
not of the merge.** `verify_merged.py` compared greedy token ids exactly between the
adapter and the merged checkpoint. The adapter computes `Wx + B(Ax)·scale` per token; the
merged checkpoint computes `(W + BA·scale)x` once. Equal in exact arithmetic, different in
bf16 — so the two argmaxes agree except where the reference's own top two logits are
within a hair, and one flip rewrites the context for every later token, making every
downstream comparison answer a different question. The old test demanded bit-exact
argmaxes at a resolution the arithmetic cannot hold.

The rewritten instrument is teacher-forced and reads its result against its own
resolution: the same forward twice (the noise floor), then adapter against merged (the
effect). Measured, on both prompts:

| | max\|dL\| | mean\|dL\| | KL | top-1 agreement |
|---|---|---|---|---|
| **noise floor** (adapter vs adapter) | **0.00000** | 0.000000 | **0.000e+00** | — |
| effect (adapter vs merged), 1261 pos | 4.67188 | 0.030748 | 2.289e-03 | 1243/1261 (98.57%) |
| effect (adapter vs merged), 1764 pos | 0.84375 | 0.028421 | 1.338e-03 | 1734/1764 (98.30%) |

The floor is **exactly zero**, so the instrument's resolution is not merely small — a
repeated forward pass is bit-identical. Against that, the effect is real and small: the
merged checkpoint's next-token distribution differs from the adapter's by KL ~2e-3 while
choosing the same token at 98.5% of positions.

The sharper evidence is the mismatches themselves, and it is visible in the printed
logits rather than in any threshold. Every one of the 48 sit at a reference margin of
**0.00000, 0.0625, 0.125 or 0.25** — and those are exactly **bf16 ULPs at the local logit
magnitude** (2^(4−7) = 0.125 at 16, 2^(3−7) = 0.0625 at 15.8, 2^(5−7) = 0.25 at 32). The
model's logits *are* bf16, so the margin between the top two can be zero, and **six of
the 48 are at margin exactly 0.00000** — literal ties, where the reference logits are
byte-identical and the tie carries no information about which token is more likely. That
is the whole defect, stated as a number: at this resolution, a greedy comparison across
two arithmetic orders is not a strict test that happened to fail; it is a test that
*defines* a coin flip as a failure.

The derived bound is `2 × max|dL|` — a flip needs `|d(top1) − d(top2)| > margin` and no
per-logit delta can exceed `max|dL|` — and it is sound but loose: 9.34 allowed against the
0.25 actually observed. It is kept because it is derived rather than hand-picked, but a
future reader should not mistake "inside the bound" for "explained"; the margin
distribution is what explains these.

One number is worth flagging rather than smoothing over: `max|dL| = 4.67` is *not* a
rounding-sized difference at that single position, and it is 22× the previous
last-position-only measurement (0.211). It is consistent with bf16 error in the merged
*weights* being amplified by a large-activation token, and it moves no verdict — the
agreement rate and the KL are computed over all 1261 positions and are unaffected by one
worst case. It is recorded because the mean (0.031) and the max (4.67) differ by two
orders of magnitude, and only one of those numbers describes the merge.

**A latent bug in the rewrite, caught by reading rather than by running.**
`verify_quant.py` imports `greedy` from `verify_merged`. The rewrite deleted that
function. Nothing imports it until step 6's quant check, so the failure would have landed
*after* a merge, a two-minute verification and a full GGUF conversion — the same shape as
every other late, wrong-reason failure this project has paid for. `greedy()` is restored
beside the loaders it shares a decode path with, and the call signature is now proven by
actually performing the import rather than by inspection.

**Step 6, stages 3-5: the deploy chain reported success on a model that could not answer
one request.** The chain exited 0. `llama-quantize` quantized 320/320 tensors and
reported `quant size = 1204.91 MiB (5.37 BPW)`. `ollama create` printed `success` and
`ollama show` reported a 1.9B `qwen35` model at Q4_K_M. The first real request returned:

```
HTTP 500  error loading model: check_tensor_dims: tensor 'blk.24.attn_norm.weight' not found
```

The converter computes `block_count = num_hidden_layers + mtp_num_hidden_layers` unless
it is told the MTP head is absent, and it only decides that for itself when
`text_config.architectures` names `Qwen3_5TextModel` (`conversion/qwen.py:674`). This
checkpoint's config has `text_config.architectures = None` — **in the base as well as in
the merge**, so this is not something the merge corrupted; the base is simply the case
where the arithmetic is honest, because the base *ships* its 15 `mtp.*` tensors and the
converter writes them as `blk.24.*`. The merge drops them (`AutoModelForImageTextToText`
does not own `mtp.*`, which stage 1 already reported). So the GGUF declared a 25th block
whose 15 tensors did not exist.

**Why two stages called that a success.** Neither builds a compute graph.
`llama-quantize` walks the tensors that are in the file — 320 of them, ending at
`blk.23.post_attention_norm.weight` — and never asks whether the metadata implies more.
`ollama create` copies the blob and reads its metadata. The missing tensor is only
noticed at `llm_load_tensors`, i.e. at the first token. A load failure is therefore
**invisible from the GGUF side**, which is the property that made this silent, and it is
the same shape as the template defect and the `verify_merged` rewrite: a check that
passes for a reason unrelated to what it claims to check.

**The fix, and why it is the right direction rather than restoring the tensors.**
`--no-mtp` is threaded through `run_convert.py` and passed by `deploy_local.sh`. The
re-converted file reports `qwen35.block_count = 24`, no `nextn_predict_layers` key at
all, `blk.23` present and `blk.24` absent — metadata consistent with tensors. The
alternative was to copy the 15 `mtp.*` tensors from the base into the merged checkpoint,
which would make `block_count = 25` honest and would even leave a usable speculative
draft head. It was rejected because nothing trained that head toward summarization, it
costs ~118 MB in f16, and the constraint this build is under is the opposite of that —
a head that is never decoded is weight to delete, not weight to preserve. The MTP drop
was already *reported* at merge time; reporting it turned out not to be the same as
*acting* on it downstream, which is the actual lesson here.

**The chain now ends with a generation, not a creation.** `smoke_deployed.py` is stage
6: it asks the deployed model for 16 tokens through `/api/generate` and fails on an HTTP
error, on an empty response, or on `eval_count == 0`. It is deliberately cheap — a short
prompt, seconds not minutes — and it is placed after `ollama create` because that is the
only point where the question "does this actually run" can be asked at all. On the fixed
artifact it prints `PASS: the deployed model loads and generates tokens`, and a real call
through the student's own template returns a summary of 91 tokens that names the
counterparty, both dates, and the outstanding deal.

**The quantization-parity instrument failed a healthy model.** The rewritten
`verify_quant.py` was built to the three-distance design: `bf16 HF vs f16 GGUF` as the
engine floor, `bf16 HF vs Q4_K_M GGUF` as engine + quantization, `f16 GGUF vs Q4_K_M
GGUF` as quantization alone, with the verdict relative to the measured floor rather than
to an absolute threshold. Run against the deployed artifact it reported a median
character-prefix agreement of **0.3215 for the f16 control** and **0.0251 for Q4_K_M** —
the quantization-free model sitting *13× further* from the bf16 reference than the
quantized one. By its own rule (`floor > 0.2 and got < 0.5 * floor`) that is a FAIL, and
it printed one.

**The recorded texts refute the verdict, which is why the texts were recorded.** For the
same prompt the f16 GGUF opens *"Rep checked in with Sam Whitfield of the fund
regarding plans for the remainder of the year, which…"* — near-verbatim the bf16
reference — while Q4_K_M opens *"Rep followed up with Sam Whitfield regarding planned
auction trips in Plano…"*. Both are grounded summaries of the same call, 496 characters
against a 500-character reference. The quantized model does not corrupt; it picks a
different near-tied continuation within the first ~10–40 characters and then decodes
independently, which is exactly what a 4-bit weight perturbation does to an early argmax.

**So prefix agreement measures trajectory, not quality, and one early argmax flip
destroys it.** The rule's premise — "diverges in the first few characters ⇒ the state
tensors were wrecked" — is false for a model whose failure mode is paraphrase: the same
signature is produced by a healthy model that chose a synonym. This is the second
consecutive failure of this instrument's verdict logic, and it retires the instrument for
this question instead of re-tuning it. The quantization question is now open in the other
direction: **the deployed model's quality is still unmeasured**, and the only way to
measure it is a paraphrase-tolerant metric on the deployed model's *own* outputs — ROUGE
against the reference, ungrounded names and numbers, copy rate, empties, mid-sentence
ends. `measure_local.py` produces that sample; scoring it is the remaining step.

**`measure_local.py` reported a prefill this CPU cannot do.** It asked Ollama to hold the
model for 30 minutes and never accounted for what that buys: Ollama reuses the KV cache of
an *exact* prompt across requests and across processes, so a prompt an earlier run had
already sent returns `prompt_eval_duration` near zero — 0.07–0.12 s for 626–1,764 tokens,
an implied 10,000+ tok/s. Cross-checked against `llama-bench` on the same file
(`pp512` 273, `pp1764` 206, `tg128` 21.8 tok/s) the field is wrong by ~60×, and the tool's
headline — *"median call 4.0s, 3% of it prefill"* — described three prompts the
`verify_quant` run had cached sixty seconds earlier. Fresh prompts tell the truth:
933 tokens → 5.02 s prefill (186 tok/s), 1,764 tokens → 11.2 s (157 tok/s), with decode
agreeing with `llama-bench` all along (21.4–24.0 vs 21.8 tok/s). Default is now
`keep_alive=0` so every call pays its own prefill, `--keep-alive` restores the cached
behaviour, and a guard flags any prefill implying more than 2,000 tok/s as a cache hit
rather than a measurement. **The shape of the mistake is the same as the other three in
this section** — the template directive, the MTP metadata, the merged-logits check, and
now this: a number read from a field rather than from physics. If the implied throughput
is impossible for the hardware, the field is wrong, not the hardware.

**The RK3588 latency projection is retired, in shape and in magnitude.** D9's open risk
("prefill dominates CPU latency") and the "Latency risk, corrected" entry above both rest
on RK3588 figures — ~51.5 s prefill / ~22.0 s decode at the median, ~74 s per call, ~70%
prefill — that were never a measurement of the deploy target. On this laptop's CPU
(Intel Core Ultra 9 285H, 16 threads, 100% CPU confirmed via `ollama ps`, no iGPU
offload) the measured figures over all 144 calls are a median 16.0 s per call at 67%
prefill — p90 26.6 s, max 33.0 s — at 163 tok/s prefill and 27.7 tok/s decode.
The *shape* survived — prefill is still the larger half — but the magnitude was ~5×
optimistic about the hardware, and "prompt length is a first-class lever" is now measured
on the target rather than inferred from a Raspberry-Pi-class board.

