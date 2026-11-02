# Knowledge Distillation for Summarization — Resources

Curated for this mission only. Knowledge is drawn from here, not from memory.
Last verified: 2026-10-13.

## Knowledge

### The method that applies to you (sequence-level)

- [Paper: Kim & Rush, "Sequence-Level Knowledge Distillation" (EMNLP 2016)](https://arxiv.org/abs/1606.07947)
  · [ACL Anthology](https://aclanthology.org/D16-1139/) · [slides](https://nlp.seas.harvard.edu/slides/emnlp16_seqkd.pdf)
  **The primary source for this entire mission.** Defines Seq-KD: run the teacher's
  beam search over the training inputs to make a pseudo-labelled dataset, then train
  the student by ordinary cross-entropy on it. Reports ~10× faster students with
  minimal loss, and that Seq-KD removes the need for beam search at *test* time.
  Use for: why beam search for the teacher, why the student can decode greedily.
  **Note what this paper does *not* settle (measured in this project, 2026-10-13):**
  "beam search is not needed at test time" is a claim about *quality*, and it left the
  *sampling parameters* unexamined. Running the shipped artifact greedily held ROUGE-1
  level while cutting ungrounded names from 37 to 24 and ungrounded numbers from 105 to
  76 — so the decoding configuration was moving a quantity the mission cares about, and
  the choice between the two is a measured trade rather than a formality. See
  [lessons/0006-the-knob-you-never-turned.html](lessons/0006-the-knob-you-never-turned.html).

- [Paper: Holtzman, Buys, Du, Forbes & Choi, "The Curious Case of Neural Text
  Degeneration" (ICLR 2020)](https://arxiv.org/abs/1904.09751)
  **The primary source for lesson 06.** Shows that greedy and beam decoding of a language
  model produce degenerate, repetitive text, and that the fluency people attribute to
  sampling comes from *truncating the tail* (top-p / nucleus sampling) rather than from
  randomness as such. Use for: why sampling is the default at all, and why "greedy is
  cheaper but worse" is the field's prior — a prior this project's run then measured
  rather than assumed, finding the ROUGE difference a tie and the grounding difference real.

- [Paper: Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)
  The origin of soft targets and temperature. Included so you know what you are
  *not* doing: it requires teacher and student to share an output space, which is
  exactly what a different tokenizer destroys. Use for: the general framing, and to
  understand why the shared-vocabulary assumption is the load-bearing one.

### When the simple recipe is not enough

- [Paper: Agarwal et al., "On-policy Distillation of Language Models" / GKD (2023)](https://arxiv.org/abs/2306.13649)
  Frames distillation as imitation learning and fixes the train/inference
  distribution mismatch by having the *student* generate sequences that the teacher
  then scores. Interpolates between off-policy SFT (`lmbda=0`) and fully on-policy
  (`lmbda=1`). Use for: reading when off-policy distillation plateaus.

- [Paper: Gu et al., "MiniLLM" (ICLR 2024)](https://arxiv.org/abs/2306.08543)
  Argues forward KL is the wrong objective under capacity mismatch — it forces the
  student to cover regions of the teacher's distribution it cannot represent — and
  substitutes reverse KL. Use for: understanding *why* a smaller student degrades,
  not just that it does.

- [Paper: "Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes" (2026)](https://arxiv.org/html/2603.25562v2)
  Names three concrete failure modes, one of which is **tokenizer / special-token
  mismatch** — directly relevant here, since teacher and student share no
  special tokens either. Use for: debugging a distillation run that trains but
  does not converge.

### The trap specific to summarization

- [Paper: "Enhancing Abstractiveness of Summarization Models through Calibrated Distillation" / DisCal (2023)](https://arxiv.org/html/2310.13760)
  Shows that naive sequence-level distillation **reduces abstractiveness** by
  worsening copy bias, and fixes it by ranking diverse pseudo-summaries on both
  informativeness (ROUGE) and abstractiveness (novel n-grams). Use for: the single
  most likely way your distilled summarizer gets quietly worse. Your project
  already measures copy rate, so this is testable rather than theoretical.

### Reading a training run (lesson 03)

- [Essay: Karpathy — "A Recipe for Training Neural Networks" (2019)](https://karpathy.github.io/2019/04/25/recipe/)
  **The primary source for lesson 03.** Its first instruction is to become one with the
  data and stand up a dumb baseline before anything clever, and its central habit is the
  one that lesson teaches: form an opinion about what the loss curve will do *before* you
  look at it, so that a surprise is information rather than a story you tell afterwards.
  Use for: the discipline of treating a run as an experiment you are predicting, and for
  the "verify the loss at init / overfit a single batch" sanity steps.
  See [lessons/0003-reading-a-loss-curve.html](lessons/0003-reading-a-loss-curve.html).

- [Paper: Zhang, Bengio, Hardt, Recht & Vinyals, "Understanding Deep Learning Requires Rethinking Generalization" (ICLR 2017)](https://arxiv.org/abs/1611.03530)
  The extreme version of the point: networks fit **random labels** perfectly and still
  drive training loss to zero, so capacity to memorise is not bounded by the classic
  measures. Included so "it converged" and "it will work" are never confused.
  Use for: how far train loss can sit from usefulness in principle.

- [Guide: Google — "Deep Learning Tuning Playbook" (Godbole, Dahl et al.)](https://github.com/google-research/tuning_playbook)
  High-trust, opinionated, and specific about the thing lesson 03 turns on: which knobs
  to touch in what order, how to read train/val curves to choose between configurations,
  and why a single run's val loss is a weak signal. Use for: what to change *next* — early
  stopping, schedule shape, and how to decide a run is not worth continuing.

### Practical tooling

- [Docs: Hugging Face TRL — Distillation Trainer / GKDTrainer](https://huggingface.co/docs/trl/distillation_trainer)
  The maintained implementation of on-policy KD, with PEFT/LoRA support and
  `lmbda`/`beta` knobs. Note the documented limits: `lm_head` adapters and
  prompt-tuning are rejected. Use for: lesson 3 onward, if you go past plain SFT.

- [Docs: Hugging Face LLM Course, ch. 11.4 — LoRA / PEFT](https://huggingface.co/learn/llm-course/en/chapter11/4)
  LoRA hyperparameter guidance (rank, alpha, dropout, `target_modules`) and
  adapter merging. Use for: setting the student's adapter config.

- [Guide: Meta AI — Distillation how-to](https://developer.meta.com/ai/docs/how-to-guides/distillation/)
  Vendor-neutral walkthrough of the synthetic-data workflow: curate inputs →
  teacher generates → baseline eval → fine-tune student → re-eval. Also the
  clearest taxonomy of distillation signals (hard outputs / logits / hidden
  states). Use for: the end-to-end shape of the pipeline.

- [Model card: Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
  Authoritative on the student: 2B params, hidden 2048, 24 layers, vocab **248,320**,
  hybrid Gated DeltaNet + Gated Attention, 262K context, Apache-2.0, **non-thinking
  by default**, and a stated warning that the 2B is *"more prone to entering
  thinking loops."* Use for: chat template, sampling parameters, and sizing.

- [Docs: Hugging Face — Xet, our storage backend](https://huggingface.co/docs/hub/storage-backends)
  The Hub's two storage backends: Xet (chunk-level deduplication) alongside legacy
  Git LFS. **The load-bearing detail:** a tree entry carries *two* hashes for one
  file — `lfs.oid`, the sha256 of the bytes, and `xetHash`, Xet's own content hash.
  Verifying against the wrong one makes a byte-perfect download look corrupt, and the
  CDN's `ETag` derives from the Xet path so it agrees with the wrong one. Use for:
  verifying the student's base checkpoint, and any future large fetch. See
  [reference/verifying-model-downloads.html](reference/verifying-model-downloads.html).

## Wisdom (Communities)

- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
  Best signal on CPU inference, GGUF quantization, and whether a given
  architecture actually runs in llama.cpp yet. Use for: the CPU-only constraint,
  where the docs lag the code.

- [Unsloth Discord](https://discord.gg/unsloth) and the [Hugging Face Discord](https://hf.co/join/discord)
  Practitioners doing exactly this LoRA-SFT-on-Qwen work. Use for: training runs
  that fail in ways the docs do not cover.

## Gaps

- **The teacher's number is a fit number and the student's is a held-out one.** The mission's
  distillation split (`finetune/teacher_input.jsonl`, 154 calls, 142 train / 12 val) has **zero**
  overlap with the 144-call eval set — verified by matching each eval call's `model_input`
  against every training row, not by trusting a split flag. The student's 0.4399 is therefore
  measured on calls it never saw. The same test against the 291-row corpus in
  `finetune/{train,val}.jsonl` finds **137 of the 144** eval prompts present, so most of the eval
  set sat in the training pool that produced the 8B. That makes the headline gap *conservative
  against the student*: it is being compared with a teacher that had seen 95% of the test set
  while it had seen none of it. `EVAL_FROZEN.json`'s `note` asserts all 144 were training rows
  drawn from a 305-row pool that is not on disk here, so the exact figure cannot be settled from
  what exists locally; the file is guarded by `content_sha256` and was left alone.
- No summarization-specific source on how far a 2B can be pushed before the
  500–1200 character house style breaks down. You will be generating that evidence.

### Closed

- **llama.cpp / GGUF support for Qwen3.5's hybrid Gated DeltaNet layers** — resolved by
  measurement, not by a source. llama.cpp merged native `qwen35` support (Feb 2026) and a
  fused CPU `GATED_DELTA_NET` op (Mar 2026); this project then converted, quantized and
  served the student through Ollama end to end. The residual risk moved: it is no longer
  "will it run" but "did the Gated DeltaNet state tensors survive Q4_K_M", which is why
  `finetune/verify_quant.py` exists. See
  [reference/distillation-glossary.html](reference/distillation-glossary.html), "GGUF /
  quantization".
- **The CPU throughput of Qwen3.5-2B at 4-bit on the deploy target** — measured, not cited.
  Over all 144 deliverable calls, on the Intel Core Ultra 9 285H (16 threads, 100% CPU, no
  iGPU offload), the deployed student costs a **median 16.0 s per call at 67% prefill**
  (p90 26.6 s, max 33.0 s), at **163 tok/s prefill and 27.7 tok/s decode** token-weighted.
  `llama-bench` on the same GGUF agrees to within about a third rather than exactly —
  `pp1764` 206 tok/s against 163, `tg128` 21.8 against 27.7 — which is the honest form of
  "two instruments agree": same machine, different kernel paths. Zero prompts were served
  from cache, so every one of those seconds is fresh prefill. The figure that decided the
  mission's "much lower latency" question is therefore in hand, and it retires the
  RK3588-derived projection that stood in for it. See
  [reference/distillation-glossary.html](reference/distillation-glossary.html), "Prefill vs
  decode".
- **Whether the deployed model's 0.0114 gap is the quantization or the engine** — measured, and
  the answer is the quantization alone. Step 8 compared three things at once (weight precision,
  inference engine, random draw), so its single number could not be attributed. Serving the
  **f16 GGUF through the same engine at the same seed**, over the same 144 calls in the same
  input order, splits it exactly:

  | what varies | arms | delta ROUGE-1 | 95% CI | verdict |
  |---|---|---|---|---|
  | engine + draw | 16-bit transformers − 16-bit Ollama | −0.0048 | [−0.0164, +0.0067] | tied, 65W/79L |
  | precision alone | f16 − Q4_K_M, both Ollama | **+0.0162** | [+0.0057, +0.0266] | **real**, 83W/61L |
  | **sum** | 16-bit transformers − Q4_K_M | **+0.0114** | [+0.0001, +0.0230] | step 8, reproduced |

  Two things follow. First, **the engine and the draw cost nothing measurable** — through
  llama.cpp at a fixed seed the same 16-bit weights score 0.4447 against 0.4399 inside
  transformers, while writing **16 words fewer** (125.0 against 141.2). Second, and less
  comfortably, **step 8's 0.0114 was diluting the effect it was reporting.** The engine/draw
  term happens to pull the other way by 0.0048 — inside its own noise — so the bundled number
  read as "real by a whisker" with a lower bound at +0.0001, when the quantity anyone actually
  cares about, the cost of shipping 4-bit, is **+0.0162 with a lower bound of +0.0057**. Nothing
  in step 8 was mis-measured; a number answering a question it was not designed to answer is
  precisely what the control existed to unpack. On grounding, the ordering is
  f16 writes 33, the shipped 4-bit 37 and the 16-bit transformers arm 46 ungrounded names, so the engine change improves invention
  too, and 4-bit gives a little of that back. Caveat: seed 42 fixes the *draw*, not the output —
  different logits mean the token streams still diverge, so the seed removes the draw as a
  nuisance variable rather than making the arms comparable token for token. See
  `finetune/scores_f16.json` and `finetune/scores_precision.json`.
- **Whether the invented names are a decoding artifact or a property of the weights** —
  measured, and the answer is decoding, so the fix is a flag rather than a retraining run.
  Serving the **shipped Q4_K_M artifact greedily** — same weights, same engine, same seed,
  decoding the only variable — over the same 144 calls cuts **ungrounded names from 37 on
  26 calls to 24 on 16 calls** and **ungrounded numbers from 105 to 76**, at an
  **identical length** (108.7 words against 108.0), which rules out "a shorter summary has
  fewer chances to invent." On quality, the honest reading is narrower than the raw
  numbers: **ROUGE-2 is a real gain** (+0.0119, [+0.0035, +0.0193], 91W/53L) while
  **ROUGE-1 (+0.0038, [−0.0101, +0.0170]) and ROUGE-L (+0.0093, [−0.0003, +0.0174]) both
  tie**, so what is established is *"greedy is not worse on ROUGE"* — no metric moved
  down — with the improvement claim resting on ROUGE-2 plus the grounding counts, and the
  grounding counts have **no paired test**. Two caveats travel with it. The reduction is
  **net, not per-call**: sampling invents more names on 19 calls, greedy on 8, 117 tie, so
  greedy's 24 are a partly *different* set rather than a subset. And the cost is real on
  both sides — one of the 144 greedy calls hit the 900-token cap and ended mid-sentence
  where the sampled run had none (greedy cannot escape a loop it enters), copy rate rises
  slightly (0.0158 against 0.0117, the direction Seq-KD is documented to push), and it
  costs **1.16× the wall time** (median 18.5 s against 16.0 s; 1.07× prefill, 1.09×
  decode). What it buys is determinism: the seed-dependence that made step 9's arms hard
  to compare disappears as a *deployment* concern. See
  `finetune/scores_greedy.json` and `DESIGN.md` step 10.
- **Whether 8-bit quantization is the better operating point than 4-bit** — measured, and
  the answer is no, for a reason nobody predicts from the size ratio. On quality, **Q8_0's
  cost against f16 is not detectable**: all three ROUGE variants tie (+0.0058 [−0.0038,
  +0.0157] at ROUGE-1) while 4-bit's cost is real (+0.0162). On speed it loses badly:
  **32.4 s median per call, 2.02× the shipped 4-bit**, with token-weighted prefill of
  **82.2 tok/s against f16's 101.7** — a model nearly twice its size is faster at prefill.
  A separate `llama-bench` run reproduces the ordering (pp512 **291 / 98 / 115** tok/s,
  tg128 **29.8 / 17.4 / 13.2** for Q4_K_M / Q8_0 / f16). **The transferable point is that
  quantization is not a monotonic speed/quality dial**: the cost of the last halving of
  bit width is where nearly all of 4-bit's quality loss lives, and the *middle* operating
  point is the slowest of the three because that is where the backend's kernels are
  weakest. Choose a quantization by measuring it beside its neighbours, never by
  interpolating. See `finetune/scores_q8_vs_f16.json`, `finetune/scores_q8_vs_q4.json`,
  `finetune/bench_precision.log`, and `DESIGN.md` step 11.
- **Whether giving the student call metadata improves its summaries** — measured, and the
  answer is "yes on grounding, no on overlap." Two 2B LoRAs, same base, recipe, seed and
  284 rows, differing only in the user message: `plain` reads the transcript alone, `meta`
  reads the prospect name, deal stage and deal outcome above it. **All three ROUGE variants
  tie** (+0.0058 [−0.0041, +0.0157] at ROUGE-1, 76W/68L), while ungrounded names fall from
  **99 on 51 of 144 calls to 39 on 23** — a diffuse reduction, not an outlier. Against the
  currently shipped `student-blend` the metadata arm also ties on ROUGE and edges ahead on
  names (39 vs 46). So metadata buys factual discipline rather than reference overlap, and
  a ROUGE-only evaluation would have declared the two arms identical. **Method note that
  cost this run a false result:** grounding each arm against its *own* prompt reported
  plain at 117 names, mostly because a stage inferred correctly from the conversation was
  absent from plain's corpus. `score.py --corpus variant=field` now grades both arms
  against the same fullest factual record; use it for any A/B whose arms read different
  prompts. Deal *size* could not be tested — `entities` is empty for 289 of 305 calls. See
  `finetune/scores_meta_ab_samecorpus.json`, `finetune/build_meta_ab.py`, and `DESIGN.md`
  step 12.
- **Whether the metadata model is worth deploying** — measured end to end on the deploy
  target, and it is the first change in the project to beat the shipped artifact on all
  three ROUGE variants at a real paired difference. Merged, converted, quantized to
  Q4_K_M (1.27 GB, same footprint) and served through Ollama with identical parameters,
  it scores **0.4406 ROUGE-1 against the shipped 0.4285** — **+0.0121 [+0.0020, +0.0221]**,
  with ROUGE-2 +0.0110 and ROUGE-L +0.0088 also real, and 32 ungrounded names against 37
  and 82 numbers against 105. The cost is **1.26× the latency** (20.2 s against 16.0 s),
  which is a far better trade than the 16-bit fallback's 1.85× for +0.0162: the deployed
  4-bit metadata model now sits within 0.0041 of f16 at two-thirds of f16's speed cost.
  **Two limits, stated so they are not rounded up:** the arm is *given* the prospect name,
  deal stage and outcome, so part of the gain is conditioning rather than a better model;
  and step 12's controlled A/B found metadata alone ties on ROUGE, so the claim is
  "metadata plus this run beats the shipped blend", not "metadata explains 0.0121". See
  `finetune/scores_meta_q4_vs_shipped.json`, `finetune/local_samples_meta_q4.jsonl`, and
  `DESIGN.md` step 13.
- **Whether a second target output needs its own model** — measured on Next Steps, and the
  answer is "only for shape, not for overlap." Labels were extracted from existing text
  (the `Next step(s):` sentence in the corpus, teacher fallback), giving 116 training
  calls and 75 benchmark calls with a human reference. A 2B LoRA against the shipped
  summarizer prompted zero-shot: **ROUGE-1 and ROUGE-2 tie** (−0.0139 [−0.0374, +0.0091]
  and −0.0063 [−0.0206, +0.0086]), while **ROUGE-L is real** (+0.0270 [+0.0090, +0.0460],
  47W/28L). The LoRA writes **32.6 words against 65.8** with **0 ungrounded numbers
  against 20** — a sentence rather than a briefing, which is the right shape for a CRM
  field. **Two of the pilot's own defects are part of the finding:** the label extractor
  initially ate the sentence terminator, so the first run emitted 75/75 outputs ending
  without punctuation; and a partial baseline join (74/75 after an Ollama HTTP 500) was repaired rather
  than accepted, because a short join moves the mean silently. Caveats: 116 rows is a third
  of the summarizer's training set, and the training calls are far shorter than the eval
  calls (251 vs 1,034 words), so this is partly a generalisation test. See
  `finetune/scores_nextsteps_pair.json`, `finetune/build_nextsteps.py`, and `DESIGN.md`
  step 14.
- **Whether 6-bit quantization is the better operating point** — measured, and no: it is a
  tie at 1.65× the cost. Q6_K, built from the deployed model's f16 GGUF, scores
  **0.4415 ROUGE-1 against Q4_K_M's 0.4406** with all three paired intervals crossing zero
  (−0.0009 [−0.0114, +0.0095] at ROUGE-1), while the median call goes from **20.2 s to
  33.4 s** and the file from 1.27 GB to 1.56 GB. `llama-bench` independently shows the same
  kernel effect that sank Q8_0: Q6_K's prefill is **193.6 tok/s** against Q4_K_M's 255.9
  and **Q5_K_M's 272.1**. The public card that recommends Q6_K
  (`mradermacher/Mixture-Summarizer-Qwen3.5-2B-GGUF`) is describing quality-per-bit, which
  is accurate; the shipping decision here is quality-per-second, and a tie at 1.65× is not
  a trade. **Q5_K_M is the only quant measured that is faster than Q4_K_M at prefill**
  (272.1 vs 255.9) at 8% more decode cost, so it — not Q6_K — is the next candidate if a
  higher-quality point is wanted. See `finetune/scores_meta_q6_vs_q4.json`,
  `finetune/local_samples_meta_q6.jsonl`, and `DESIGN.md` step 15.
- **A metric for quantization damage that survives paraphrase.** The missing instrument was
  built rather than argued for: ROUGE against the reference, plus ungrounded names and
  numbers, copy rate, empties and mid-sentence ends, run on the deployed model's own 144
  outputs through the same scorer and the same references that produced every other number in
  this project (`finetune/scores_q4km.json`). What it shows: the deployed arm scores **0.428
  ROUGE-1 against the 16-bit arm's 0.440**, a gap the paired bootstrap calls real on all three
  ROUGE variants (+0.0114 [+0.0001, +0.0230] at ROUGE-1, 78W/66L for the 16-bit side). It is
  not the control arm's failure mode again — the deployed model's ungrounded names sit on 26
  of 144 calls, at most 4 on any one call, rather than piling 443 of the control's 512 onto a
  single runaway loop — so this is a diffuse regression, not an outlier. **One number in this
  entry was wrong when it was first written (corrected 2026-10-13):** it read "324 ungrounded
  names spread over 139 of 144 calls". That came from a scorer lookup, not from the model —
  the deployed arm's variant name was never added to `score.py`'s `COPY_CORPUS` registry, so it
  was grounded against raw ASR with anonymised speaker labels, in which the call participants'
  own names do not appear. Against the corpus the model actually read the figure is **37 on 26
  calls**, which also means the deployed 2B invents fewer names than its teacher (109), the
  untuned 8B (107) and its own 16-bit self (46). ROUGE never read that corpus, so the 0.428 and
  the paired interval above are unchanged and the re-run reproduces them to the digit. The
  registry now refuses to guess. See
  [learning-records/0002-a-quantized-model-fails-by-omission.md](learning-records/0002-a-quantized-model-fails-by-omission.md).
  This retires `verify_quant.py`'s
  prefix-agreement FAIL, which its own recorded texts already contradicted: both arms wrote
  the same call and the same facts, and one early argmax flip is enough to destroy a character
  prefix without touching quality. Nothing found in the literature states what a healthy
  cross-engine prefix agreement looks like, and this run is evidence that building such a norm
  is not worth the effort when a paraphrase-tolerant metric is available.
- **Whether the 16-bit model is a viable fallback if quality had demanded it** — measured on
  this hardware over the full 144 calls, and the answer is no — but not for the reason this
  entry first gave. On the **142 calls neither run slept through**, f16 costs **1.85× the wall
  time per call** (median 29.0 s against 15.7 s; over all 144 the deployed arm's median is
  16.0 s, p90 26.6 s, max 33.0 s). The split is the informative part: **prefill 1.54×**
  (15.9 s against 10.3 s) and **decode 2.51×** (11.8 s against 4.7 s). Prefill sits far below
  f16's own 2.96× size ratio and decode sits just under it, which is what a compute-bound
  prefill beside a bandwidth-bound decode should look like on this class of CPU — the entry's
  original thesis, now with the numbers behind it. **Corrected 2026-10-13:** it first read
  1.71× prefill / 1.94× decode at a median 28.2 s against 16.1 s, computed while the f16 run
  was still in progress. The full run is 1.54× / 2.51× at 29.0 s against 15.7 s. The conclusion
  is unchanged; the evidence for it is stronger.
  **The clause "for no gain in quality" was also wrong, and step 9 is what proved it.** The
  fallback does buy something real. Served through the same engine at the same seed, f16 scores
  **0.4447 against the shipped 4-bit's 0.4285** — **+0.0162 ROUGE-1**, a difference the paired
  bootstrap calls real on all three variants (+0.0057 to +0.0266 at ROUGE-1). So this is a
  genuine trade rather than a free lunch: a **3.8% relative** quality gain for 1.85× the time on
  every call, on a machine whose user is waiting. The answer stays no, but it now rests on the
  mission's own priority — runs locally without using too many resources — instead of on the
  claim that nothing is purchased.
  **A second argument — that the f16 mapping is evicted under memory pressure while the
  1.27 GB artifact keeps its pages — was written here and has been withdrawn (2026-10-13).**
  It rested on a single 1,950 s f16 row read as a paging collapse. The Windows event log
  (`Microsoft-Windows-Kernel-Power`, IDs 506/507, Modern Standby) instead shows a **1,936 s
  suspend** across exactly that decode, entered *after* decoding had begun; the run's second
  outlier ends at the second of a logged 09:35:12 wake. The machine slept — nothing was
  evicted. The q4km run, the one that ships, has **no row over 40 s at all**, so no published
  latency figure is affected either way. See
  [lessons/0005-ship-it-then-check.html](lessons/0005-ship-it-then-check.html), "What this buys
  the mission".
