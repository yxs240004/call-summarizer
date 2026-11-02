# An outlier is a measurement, not an explanation

Established 2026-10-13. The f16 arm produced one row at **1,950 seconds of decode for 211
tokens** — an apparent 250× collapse to 0.108 tok/s, far outside anything else in the run. I
wrote an explanation for it and shipped that explanation into three documents: memory
pressure. Ollama memory-maps the GGUF, a 3.78 GB mapping is a large trimmable target the
moment the system wants memory back, the 1.27 GB Q4_K_M artifact keeps its pages, therefore
the smaller file is not merely faster but *more robust* — "a second, independent argument for
quantizing," as `RESOURCES.md` put it. The user's laptop had a commit charge of 30.3 GB of
32.1 GB at the time, which made it read as confirmed.

**It was wrong, and the check that refutes it costs thirty seconds.** Windows logs its own
power transitions. `Microsoft-Windows-Kernel-Power`, IDs 506 (enter) and 507 (exit) Modern
Standby, show a suspend window of **1,936 seconds** spanning exactly that decode — entered
*after* the call had already begun — and a second outlier in the same run that ends on the
second of a logged wake at 09:35:12. The machine slept. No page was evicted; the f16 model was
never the fragile one.

**The diagnostic that should have caught it first is the row's own signature.** The two
outliers in that file do not look alike: the first inflated *decode* (1,950 s, prefill 32 s),
the second inflated *prefill* (42,806 s, decode 15 s). A single-cause story has to explain
both. Page eviction against a memory-mapped weight file hits the phase that streams weights,
not the compute-bound one, so it cannot be the story for the prefill-inflated row — whereas a
suspend can land in either phase and so produces either signature. Two different shapes, one
cause. Whenever an explanation covers one outlier and quietly ignores the second, it is a
hypothesis wearing a conclusion's clothes: **read every outlier before naming the cause, and
check whether the mechanism you are proposing can produce all of them.**

**The transferable lesson:** an instrument cannot explain something it cannot see. An
application-level timer records wall-clock seconds; it has no way to know the operating system
stopped running the process, so any cause it *suggests* is a hypothesis about a layer below the
one being measured. Before attributing an anomaly to the system you are instrumenting, query
the layer underneath it — here, the OS's own record of what it did to the machine. Note too
that the wrong explanation was the *appealing* one: it made the measurement into an argument
for a decision already taken. That is what made it cheap to believe and expensive to correct
across four files.

**What survives, and what did not.** The size-and-speed argument is untouched and independently
measured: f16 costs **1.85× the wall time per call** (a median 29.0 s against 15.7 s), of which
**1.54× is prefill and 2.51× decode**, against a 2.96× size ratio — prefill sitting well below the
size ratio because it is compute-bound, decode just under it because it is bandwidth-bound. That
is still a reason to quantize — a person waits on every call — but it is a cost argument, not a
durability one. **Corrected 2026-10-13:** this paragraph first read "1.71× in prefill and 1.94× in
decode (median 28.2 s per call against 16.1 s)", computed while the f16 run was still in progress.
The full run is 1.85× / 1.54× / 2.51× at 29.0 s against 15.7 s; no method tested — median of
medians, means, token-weighted rates, or paired per-call ratios — reproduces the partial-run
figures. The conclusion is unchanged and the evidence for it is stronger. The published latency figures are also unaffected: the
shipped Q4_K_M run of 144 calls has **no row over 40 s at all** (max 33.03 s), so nothing in the
delivered numbers was ever contaminated by the suspend. Only the *explanation* is retracted.

**The shape recurred four hours later, and this time the pre-check ran first.** The greedy
run finished the same day with a single 8,088 s row (`call_id 112675989088`) that is
**8,080.5 s of prefill** against a 5.2 s decode. Before writing anything about it, the power
log was queried: `Microsoft-Windows-Kernel-Power` 506/507 show **Modern Standby from
10:03:36 to 12:18:01 = 8,065 s**, matching the prefill to within about fifteen seconds, and
the call began just before the suspend. The whole run cost 181.6 min instead of the ~75 min
its own median predicts, entirely from that one row; on the 143 calls neither run slept
through the comparison is clean. So the transferable habit is not only "check the layer
underneath" but **"check it first"** — the query costs seconds and it is what turns a
1.4-hour mystery into a footnote, and, unlike last time, nothing had to be retracted.

**This is the second time in one project that a figure answered a question it was not designed
to answer** — see [[0002-a-quantized-model-fails-by-omission]], where the same error shape
appeared as an unattributed three-way confound. The pattern is worth watching for: both times
the number was real and the story attached to it was not.

**Implications:** lesson 05 carries the retraction inline rather than being quietly renumbered,
because "know what it would have taken for it to be wrong, and go and look" is that lesson's own
thesis, and this is its thesis applied to its author. `DESIGN.md` step 10 keeps the
don't-run-two-Ollama-jobs rule but no longer justifies it with this event — the rule is about
comparability, not about the stall. `NOTES.md` should carry the power-log query as a standing
pre-check before any latency outlier is attributed.
