# Decoding is a knob, not a formality

Established 2026-10-13. The greedy run was queued for one reason and came back answering a
different, better question. Its stated purpose in `DESIGN.md` step 10 was determinism — "whether
the deployment can be made deterministic instead of seed-dependent," with the grounding question
demoted to a residual after step 8's 324 ungrounded names turned out to be 37. What greedy
actually bought was the grounding: ungrounded names **37 on 26 calls → 24 on 16 calls**, ungrounded
numbers **105 → 76**, at an **identical length** (108.7 words against 108.0). Determinism is the
real but lesser prize. The switch costs one flag, and the run was almost not worth spending an
hour on.

**Why it was almost not spent is the lesson.** Both this project's literature and its own habits
treat decoding as scenery. `RESOURCES.md`'s Seq-KD entry carries the field's framing — "Seq-KD
removes the need for beam search at *test* time" — and the sampled run had been the shipped
configuration for weeks without anyone asking what the sampling was doing. **A parameter that
appears in every deployment and in no experiment is a parameter nobody has measured.** The
sentence "we sample at temperature 0.7" reads as a description of the model. It is a description
of the *harness*, and it was moving a quantity the mission cares about by a third.

**The trap the result sets, and the discipline it demands.** Only **one of three** ROUGE variants
moved: ROUGE-2 is real (+0.0119, [+0.0035, +0.0193]) while ROUGE-1 (+0.0038, [−0.0101, +0.0170])
and ROUGE-L (+0.0093, [−0.0003, +0.0174]) both tie. The honest sentence is therefore *"greedy is
not worse on ROUGE"* — no variant moved down — and the improvement claim has to lean on ROUGE-2
together with the grounding counts, which carry **no paired test at all**. "Better on ROUGE" is an
overread of one-in-three, and a three-variant family where exactly one variant clears the bar is
the most commonly overread result in this literature. The companion trap is subtler: the counts
improved **in total, not per call**. Sampling invents more names on 19 calls, greedy on 8, and 117
tie, so greedy's 24 are a partly different set rather than a subset. A difference of totals is not
a promise about the caller's own call.

**A tie is not "no change."** Everything above sits behind the same CI-crosses-zero rule that makes
the ROUGE-1 and ROUGE-L results ties — and those ties are honest about *significance*, not about
*equality*. The right reading of a tie is "this run cannot tell you the sign," never "nothing
happened here." Which is exactly why the grounding counts, which have no interval at all, had to be
reported as counts and not converted into a verdict.

**What it cost, stated so it is not discovered later.** One of 144 greedy calls hit the 900-token
cap and ended mid-sentence where the sampled run had none — greedy cannot escape a loop it enters,
so its characteristic failure is the one sampling mostly avoids. Copy rate rises slightly (0.0158
against 0.0117), the direction Seq-KD is documented to push. Wall time is **1.16×** (median 18.5 s
against 16.0 s; 1.07× prefill, 1.09× decode). A run whose entire selling point is determinism, that
adds a truncation mode and 16% latency, is a real trade rather than a free fix.

**Implications:** the next session should test the student's decoding as an experimental variable
rather than as plumbing — the sweep that suggests itself is repetition penalty and the token cap,
since those are the two knobs the failure mode and the truncation implicate. The teaching move that
follows is to make the user *predict a per-call difference before seeing an aggregate one*, because
net-vs-per-call is the error this result invites and it generalises well past decoding. Also worth
carrying: this run is the second instance of the suspend shape in
[[0003-the-machine-slept]], and the standing power-log pre-check caught it *before* any explanation
was written — the habit, not just the knowledge, transferred.
