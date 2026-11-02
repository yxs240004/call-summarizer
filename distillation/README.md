# Distillation: a 2B summariser that replaces an 8B teacher

This folder is a self-contained record of a knowledge-distillation experiment: taking an
8B model's summaries of sales calls and teaching a **Qwen3.5-2B** student to reproduce them
on CPU-only hardware.

It is published as a worked example — the design record, the lesson material, and the
evaluation code — not as a model release. **No trained weights and no call data are in this
repository.**

## The result

| | teacher (8B LoRA) | student (2B) |
|---|---|---|
| ROUGE-1 | 0.4784 | **0.4406** (92%) |
| Size on disk | 16 GB (bf16) | **1.27 GB** (Q4_K_M) |
| Runtime | GPU | **20 s / call, CPU, 16 threads** |

The student is served locally through Ollama; the evaluation is a paired bootstrap over
the same 144 held-out calls for every arm.

## What is interesting about it

Four controlled ablations, each changing one thing:

1. **Precision.** 4-bit vs 8-bit vs 16-bit on identical weights. The cost of shipping 4-bit
   is **+0.0162 ROUGE-1**, statistically real; 8-bit's cost against 16-bit is **not
   detectable**.
2. **Engine and draw.** The same 16-bit weights through transformers and through llama.cpp:
   a tie, so step 1's number is attributable to precision alone.
3. **Decoding.** Greedy vs sampled: invented names fall from 37 to 24 at equal ROUGE-1, for
   1.16× the latency.
4. **Input metadata.** Transcript-only vs conditioned on prospect name, deal stage and
   outcome: ROUGE ties, invented names fall from 99 to 39.

A finding worth flagging for anyone quantising small models on CPU: **quantisation is not a
monotonic speed dial.** Q8_0 is *slower than f16* on this hardware, and Q6_K is 1.65× slower
than Q4_K_M for no measurable quality gain — because the backend's optimised kernels are
for the K-quant family, not for 8-bit.

## Reading order

- **`DESIGN.md`** — the decision record, all 15 steps, including the numbers behind every
  claim above.
- **`lessons/`** — six self-contained HTML lessons that teach the method end to end
  (sequence-level distillation, auditing a teacher, reading a loss curve, paired-bootstrap
  evaluation, verification, and decoding).
- **`learning-records/`** — five short records of non-obvious mistakes and what fixed them.
- **`reference/`** — the glossary and the model-download verification note.
- **`assets/`** — the shared stylesheet and quiz widget the lessons use.
- **`architecture.html`** — a diagram of how the pieces fit together.

Open any lesson directly in a browser; no build step, no server.

## What is deliberately absent

- **Weights.** The merged checkpoint, GGUF quantisations and LoRA adapters stay local.
- **Data.** The transcripts and summaries are a client's private sales calls.
- **Vendored code.** `llama.cpp` is referenced, not copied.

## Caveats a reader should carry

The evaluation is 144 calls from **one domain** on **one machine**. The teacher saw roughly
95% of the evaluation prompts during its own training while the student saw none, which
makes the headline gap favourable to the teacher — the student's 92% is conservative.
