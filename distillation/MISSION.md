# Mission: Distil the 8B call summarizer down to something a laptop CPU can run

## Why
The meeting-summarizer project produces its house-style call summaries with a
fine-tuned Qwen3-8B LoRA, which needs a 24GB GPU. The goal is the same summaries
from a model small enough to run locally on a Windows laptop with no GPU at all —
so transcripts never leave the machine, there is no per-call cost, and summaries
come back quickly. The model is only half the point: the other half is learning
knowledge distillation well enough to run it again on the next project unaided.

## Success looks like
- A Qwen3.5-2B (or smaller) student, distilled from the 8B teacher, scored on
  **held-out** calls with the project's existing scorer — not on its own training rows
- That student summarizes a call on this laptop's CPU with no GPU, in a time that
  is tolerable for how the summaries will actually be used
- You can explain unaided why this distillation is sequence-level rather than
  logit-based, and what that choice costs
- The distilled model's numbers sit next to the 8B's, including where it is worse —
  **met**: [student_report.html](student_report.html) puts the reference, the teacher, the
  16-bit student and the shipped 4-bit student on all 144 calls, with per-call ROUGE and an
  invention count, sorted worst-first by default. [summaries_report.html](summaries_report.html)
  covers the teacher-side arms on the same calls.

## Constraints
- **Deploy target is a CPU-only Windows laptop.** This is the hard wall: it decides
  model size and forces a quantized, non-PyTorch runtime — a different toolchain
  from the one used for training.
- Training hardware is unchanged — a single 24GB GPU, or a rented equivalent.
- Depth wanted: recipes and failure modes over derivations.
- The teacher currently has outputs for only ~122 of the 291 training examples, so
  teacher inference must be re-run before any student training can start.
- "Much lower latency" and "CPU-only" pull against each other, and the honest
  throughput of a 2B at 4-bit on CPU has not been measured yet. Resolving that is
  part of the mission, not a precondition for it.

## Out of scope
- Redesigning the summarization task, the house style, or the frozen eval set.
- Distilling for reasoning, agents, or any task other than call summarization.
- Multimodal work. Qwen3.5-2B ships a vision encoder; it is dead weight here.
- Chasing a particular ROUGE number as an end in itself.
