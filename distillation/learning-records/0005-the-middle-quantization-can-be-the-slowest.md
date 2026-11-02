# The middle quantization can be the slowest

Established 2026-10-22. The roadmap asked for int8, and int8 turns out to be the one
operating point on this machine that is **worse than both of its neighbours**. On quality
it is excellent — Q8_0's ROUGE against f16 is a tie on all three variants
(+0.0058 [−0.0038, +0.0157] at ROUGE-1), while 4-bit's cost is real (+0.0162). On speed it
is a disaster: **32.4 s per call, 2.02× the shipped 4-bit**, and its token-weighted
prefill (**82.2 tok/s**) is slower than f16's (**101.7 tok/s**) on weights less than half
the size. A separate `llama-bench` run reproduces the ordering (pp512 **291 / 98 / 115**
tok/s for Q4_K_M / Q8_0 / f16), so it is the backend, not the run.

**The transferable lesson: quantization is not a dial with quality on one end and speed on
the other.** It is a set of discrete operating points, and each one's cost is set by
whether an optimized kernel exists for that format on that CPU. Q4_K has heavily optimized
K-quant dot products; Q8_0 on this build does not; f16 has a plain but well-vectorized
path. So the size ratio — the thing everyone reasons from — predicts the *f16-to-Q8* gap
almost not at all, and the middle point can lose to both ends. The practical rule: pick a
quantization by measuring it **beside its neighbours**, never by interpolating between
them. A single arm compared only with f16 would have looked like a clean win.

**The second lesson is about metric corpora, and it produced a false result inside this
same session.** The metadata A/B first grounded each arm against *its own* prompt, which
reported the transcript-only arm at **117** ungrounded names against 39 — a 3× effect that
was largely an artifact. With the deal-stage line removed from that arm's corpus, a summary
that correctly *infers* the stage ("Marketing Agreement Request") is counted as fabrication.
Grading both arms against the same fullest factual record moved the figure to **99**, and
the surviving gap (99 against 39, on 51 calls against 23) is real. **An arm's grounding
corpus must contain every fact about the call, not merely the facts that arm happened to be
shown** — otherwise the A/B measures prompt formatting and reports it as truthfulness. The
scorer now exposes `--corpus variant=field` for exactly this.

**What the metadata A/B itself established:** ROUGE cannot see the difference between
transcript-only and metadata-conditioned students (all three variants tie), but grounding
can — invented names halve. A ROUGE-only evaluation would have called the arms identical
and shipped whichever happened to be cheaper. **When a change is invisible to your primary
metric, that is a statement about the metric, not about the change.**

**Implications:** the next session should resist "int8" as a default answer to a quality
worry; on this hardware the only measured improvements are 4-bit-with-greedy and
metadata-conditioned 4-bit, and both are config-level. The teaching move that follows is to
have the user *predict the speed ordering before running `llama-bench`* — the failure of
that prediction is the lesson, and it transfers to every "smaller therefore faster"
intuition they will meet. Cross-links:
[[0004-decoding-is-a-knob-not-a-formality]] for the decoding change, and
[[0002-a-quantized-model-fails-by-omission]] for the last time a metric was pointed at the
wrong corpus.
