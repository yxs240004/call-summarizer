# Prior training experience sets the floor; CPU-only sets the wall

Established 2026-10-12. The user has already trained a Qwen3-8B LoRA on this
project's data and solved a non-trivial chat-template problem unaided (the shipped
Qwen3 template defaulted thinking ON, TRL 1.x had dropped `chat_template_kwargs`,
and `assistant_only_loss` needed `{% generation %}` markers the shipped template
lacked — so they hand-rolled the Jinja). LoRA SFT, TRL, PEFT and the project's data
pipeline therefore sit **below** the teaching floor and should not be re-taught.
Claimed/evidenced depth: practitioner, not beginner.

Two facts shape every future session:

- **Deploy target is a CPU-only Windows laptop.** The distilled model must
  therefore run through a quantized, non-PyTorch runtime (llama.cpp / GGUF class of
  tooling) — a different toolchain from training, and the hard cap on model size.
- The user selected **"much lower latency"** alongside CPU-only, which are in
  tension. Measured 4-bit CPU throughput is the number that resolves it, and it can
  only come from their own hardware. See [[MISSION.md]].

**Implications:** lessons should start from a working SFT recipe they already
recognise and change one thing at a time.

**Revised 2026-10-12 (same day).** The technical risk this record originally flagged
— "unverified llama.cpp support for Qwen3.5's hybrid Gated DeltaNet layers" — is
**retired**. llama.cpp merged native `qwen35` support on 2026-02-10 and a fused CPU
`GATED_DELTA_NET` op on 2026-03-07 (+34–41% generation on Raspberry Pi 5), and both
Ollama and LM Studio ship `qwen3.5:2b`. The student's deploy path is proven, so the
mission's residual risk is *quality*, not feasibility.

**Also revised:** the original note implied latency would be decided by decode
throughput. Measured CPU numbers say otherwise — prefill dominates, though less
overwhelmingly than the longest prompt suggests. Across the 144 deliverable calls on an
RK3588, Qwen3.5-2B at 4-bit spends ~51.5 s prefilling against ~22.0 s decoding at the
median (~70% prefill, ~74 s per call), and ~95.4 s against ~31.2 s at the 90th
percentile (~75% prefill, ~127 s per call). The earlier ~112 s / ~21 s figure was the
3,620-token *maximum* prompt quoted as if typical; it overstates the median call by
about 1.5×. Prompt length is still a first-class lever on latency. There is still **no
x86 laptop measurement**; the deploy target's real number remains unmeasured.

**Revised again 2026-10-12 (later the same day): the laptop measurement exists, and it
retires the RK3588 figures above.** On the Intel Core Ultra 9 285H (16 threads, 100% CPU
confirmed via `ollama ps`, no iGPU offload) the deployed student measures, over all 144
deliverable calls, a **median 16.0 s per call at 67% prefill** — p90 26.6 s, max 33.0 s — at
**163 tok/s prefill** and **27.7 tok/s decode** token-weighted, rather than ~74 s at ~70%.
A second instrument, `llama-bench` on the same GGUF, lands in the same region but not on the
same point (`pp1764` 206 tok/s against the runtime's 163, `tg128` 21.8 against 27.7), so the
two agree to within about a third — same machine, different kernel paths. Zero of the 144
prompts were served from cache, so every one of those seconds is fresh prefill. The *shape*
of the original insight survived — prefill is still the larger half — but the RK3588's
magnitude was ~5× off for this hardware, and it was never a measurement of the deploy target
at all.

**The transferable lesson, which cost two wrong numbers:** a reported timing field is not
a measurement. `measure_local.py` first printed *"median call 4.0s, 3% of it prefill"*
because it held the model warm for 30 minutes and Ollama silently served three prompts
from a KV cache an earlier run had populated — 0.07–0.12 s for up to 1,764 tokens, an
implied 10,000+ tok/s, which no laptop CPU can do. The sanity test is arithmetic, not
trust: convert the reported duration into implied throughput and ask whether the hardware
could possibly deliver it. If it cannot, the field is wrong. The same test caught a
false FAIL in `verify_quant.py`, and the fix in both cases was to measure a
cache-free, physics-checked quantity rather than a convenient one.
