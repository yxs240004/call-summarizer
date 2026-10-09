# Glossary

Terms for the sales-call summarizer. Implementation-free — this is the shared
vocabulary, not a spec.

- **Call** — one recorded sales conversation, identified by `call_id`, stored in
  Supabase. There are two disjoint sources: the `calls` table and the
  `calls_for_analysis` view; this project reads the view.

- **Summarizable call** — a call worth summarizing: `transcription_status =
  completed` **and** `duration_seconds >= 240`. 144 such calls as of 2026-10-09.
  Shorter "completed" calls are voicemail/no-audio and are excluded.

- **Transcript** — the words spoken on a call. Exists in several forms: the raw
  ASR `transcript` (multiple JSON shapes), the `speaker_labeled_transcript`
  segments, the cleaned dialogue (`data/cleaned/<id>.txt`, fillers stripped), and
  the `model_input` (the exact user-turn text a model was trained/run on, possibly
  truncated). "Transcript" alone is ambiguous — say which form.

- **Summary** — the concise CRM-facing writeup of a call. The house style is a
  factual third-person briefing: narrative, concrete specifics, then a `Next steps:`
  sentence and an `Outcome signal:` sentence. The Supabase column is misspelled
  `summmary` (triple-m) on `calls_for_analysis`.

- **Reference summary / house summary** — the existing summaries in that column,
  written by a large LLM. They are the *style target*, not ground truth: they can
  and do contain errors. The project replaces the generator that produced them.

- **Grounded / ungrounded (fabrication)** — a figure or proper noun in a summary
  is *grounded* if it appears in the transcript the model read, *ungrounded*
  (fabricated) otherwise. The proxy that measures this over-reports; audit before
  quoting.

- **Copy rate** — share of a summary's n-grams lifted verbatim from its source
  transcript. High copy rate means the model echoes rather than summarizes.

- **Duplicate rate** — share of a summary's n-grams that repeat within the summary
  itself. High duplicate rate means the model loops.

- **Style fit vs generalization** — scoring a model on calls it was trained on
  measures style fit; scoring on held-out calls measures generalization. The two
  are different numbers and are not interchangeable.

- **Frozen eval set** — a fixed list of calls (ids + content hash) that every run
  is scored against, so numbers stay comparable across experiments. Changing it
  invalidates prior benchmarks.

- **Held-out split** — a train/eval split in which the eval calls never appear in
  training. Without it, ROUGE on the eval set is memorization, not summarization.

- **Outcome signal** — the final sentence of a house summary framing deal stage
  and engagement. It is the "answer" the model must infer from the call, so it is
  never fed in as input (unlike `stage_at_call_label`, which is).
