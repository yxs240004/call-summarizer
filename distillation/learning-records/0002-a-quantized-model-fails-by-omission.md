# The deployed model fails by omission, and invents fewer names than its teacher

Established 2026-10-12. The deployed artifact is now scored on the same 144 calls, against
the same references, through the same scorer that produced every other number in this
project. **Q4_K_M scores 0.428 ROUGE-1 against the 16-bit arm's 0.440**, and the paired
bootstrap calls the gap real on all three ROUGE variants (+0.0114 [+0.0001, +0.0230] at
ROUGE-1, 78W/66L for the 16-bit side). Real by a whisker: the lower bound sits at +0.0001,
so this is a difference 144 calls can just barely resolve, and a 145th call could plausibly
erase it.

**The failure shape is not the control's, and the total hides that.** The control arm's 512
ungrounded names were **443 from a single runaway loop** — one call, duplicate_rate 0.967,
456 words, **86.5% of that arm's entire total**. The deployed model's names do not concentrate
like that: its worst call contributes 4, or 10.8%, with **37 spread over 26 calls at a median
of 1**. Same-looking aggregate, opposite kind of failure: one is an outlier to filter, the
other is a low-rate drift to fix. Reading the totals alone would have said the two failures
were alike; the distribution says they are not.

**Corrected 2026-10-13 — this paragraph first read "the deployed model's 324 are spread across
139 of 144 calls at a median of 2".** Those numbers were a scoring artifact, not a property of
the model; the corrected figures are 37 on 26, and they reverse the sign of the comparison.
The control's 512 was never affected, because that arm was registered correctly. The cause is
in the corpus note below, and it is the more useful half of this record.

**Reading the losing calls shows the mechanism is omission, not invention.** The deployed
arm writes 33 words shorter on 110 of 144 calls. On its three worst calls it produced
80w / 70w / 130w where the 16-bit arm produced 151w / 158w / 81w against references of
181w / 171w / 113w — and ROUGE rewards recall of content, so a summary that drops a third of
the call loses heavily while inventing nothing. Where it *does* add names they are mostly
parenthetical glosses of acronyms it should have left alone: `RCFO (Rico fee…, BP303…)` where
the reference reads *Arco fee, Arco Johnson*; `Ken Shore (Kenneth Shore)` where the transcript
supports one name and the model emitted both to hedge. Some garbling is shared with the 16-bit
arm — `Nakedochis` for *Nacogdoches* appears in both — so that is inherited from the 2B, not
introduced by 4-bit.

**The limit of this record — corrected 2026-10-12.** The comparison varies three things at
once: weight precision (Q4_K_M against f16), inference engine (llama.cpp through Ollama against
transformers), and random draw (seed 42 against an unseeded `do_sample`). So 0.012 is the joint
cost of all three, and this evidence alone cannot say which dominates. **An earlier version of
this paragraph named a different confound and was wrong.** It claimed "the 16-bit arm decoded
greedily inside transformers; the served run samples at temperature 0.7." Re-opening the
generator refutes it: `student_generate.py:135` is `do_sample=True, temperature=0.7, top_p=0.9,
repetition_penalty=1.05`, the same three numbers the served run sends. Sampling was never a
confound; "greedy versus sampled" was a remembered property of a different experiment, written
into this record and then propagated into `RESOURCES.md`, `DESIGN.md` and lesson 05 before
anyone re-read the file. The error is this record's own thesis turned on its author: a figure
answering a question it was not designed to answer. Corrected limit above; the served f16
control is now the *same engine at the same sampling and seed*, which is a tighter experiment
than the one originally proposed precisely because the variable it meant to remove was absent.

**The corpus bug — why the numbers above changed, and the sharpest lesson in this record.**
`score.py` maps each variant to *the text it actually read* through a `COPY_CORPUS` registry and
looks it up with `COPY_CORPUS.get(variant, "cleaned_transcript")`. That fallback looks harmless
and is wrong for every Qwen-family arm: `cleaned_transcript` is raw ASR with anonymised
`SPEAKER_00`/`SPEAKER_01` labels, so the call participants' own names never appear in it, and
every summary that names the person on the call — which is most of them, and what the house
reference does too — reads as invented. `student_blend` and `student_control` were registered
when this was first noticed: against the wrong corpus `student_blend` scored **413** ungrounded
names; against the right one, **46**, a 9× swing produced by nothing but a lookup default.

The fix was a *name registration*. Step 8 then scored the deployed artifact under a *new*
variant name, `student_q4km`, which nobody had registered — so it fell through to the same
default and silently re-inherited the exact bug the earlier fix had closed, reporting **324**
where the matched corpus gives **37**. ROUGE never touches the corpus, which is why every
quality number in the table stayed correct while the fabrication number was wrong by 9× — and
it was wrong on the one metric the design leans on to catch what ROUGE cannot see. Two lessons,
both worth carrying: a fix expressed as "add the new name to the list" does not survive the next
name, and a default that *looks* like a sensible guess is how a silent wrong answer gets in.
The registry now stops rather than choosing (`corpus_for()` in `score.py`), so a missing entry
is a failed run instead of a plausible number.

**The confound is resolved — updated 2026-10-13.** The 46-against-37 contrast varied three
things at once, so it could not say which one cost the quality. Serving the **f16 GGUF through
Ollama at the same seed**, over the same 144 calls in the same input order, splits it exactly.
The **engine and the draw cost nothing measurable**: 16-bit inside transformers scores 0.4399
and 16-bit through llama.cpp 0.4447, a paired difference of −0.0048 with CI [−0.0164, +0.0067],
a tie. **Precision alone costs +0.0162** [+0.0057, +0.0266], real at 83W/61L. The two terms sum
to +0.0114 — step 8's published figure, to the digit.

Two corrections follow from that, and the second is the uncomfortable one. The deployed model's
loss is the quantization, not the engine. And it is **larger and firmer than the bundled number
suggested**: the engine/draw term was pulling the other way by 0.0048, inside its own noise, so
a comparison that moved three things reported "real by a whisker" with a lower bound of +0.0001
for a quantity whose true lower bound is +0.0057. Nothing was mis-measured — the number simply
answered a question it was not designed to answer. Inverted once by the corpus bug and diluted
once by the confound, the cost of shipping 4-bit settles at **0.0162 ROUGE-1**. See
[[0003-the-machine-slept]] for the other measurement defect found the same week.

**Implications:** for the mission, the deployed artifact sits at **~89% of its teacher's
ROUGE-1** (0.4285 against 0.4784) and **~96% of its own unquantized self** as served through the
same engine (0.4285 against 0.4447), which is inside "almost as good as the teacher" — with the
caveat that "almost" is now a measured, significant **0.0162** rather than an assumed small
number, and that the control which measured it has run. See [[MISSION.md]] and
`finetune/scores_q4km.json`.

**The two metrics disagree, but in the opposite direction to the one first published.** Read
on the same 144 calls, by the same scorer, against the same corpus, the 16-bit student writes
**46** ungrounded names on 30 calls and the shipped 4-bit student writes **37** on 26. Quality
moved 2.6% down; invention moved about **20% down as well**. The arm that lost precision
invented *fewer* names, not more. For scale, the teacher writes **109** on 52 calls, the
untuned 8B **107** on 31, and the tuned 4B **47** on 28 — so the deployed 2B, the smallest model
in the table, produces the fewest invented names of any model arm in it. A plausible reading is
the omission mechanism below, since a summary that writes 33 fewer words has less room to
embellish; that is a hypothesis, not something this measurement separates. An earlier version
of this paragraph reported "324 on 139" and a **7×** invention penalty. That was the corpus bug,
and it inverted the finding.

**The transferable lesson, and the reason this record exists:** a total tells you a system is
worse; a distribution tells you *how*, and the two answers lead to different repairs. And
when a comparison moves two things, the honest output is two numbers and a named confound —
never one number attributed to whichever cause you were investigating. This is the same
mistake shape as the four reported-field defects in lesson 05: a figure that answers a
question it was not actually designed to answer.
