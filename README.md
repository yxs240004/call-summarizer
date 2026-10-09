# call-summarizer

Summarizes Supabase sales-call transcripts with a small open-weight model (target ≤4B params).

## Status

- Smoke test (2026-10-09): Qwen3-8B-LoRA beats BART, ROUGE-1 **0.478 vs 0.201**, copy rate **0.01**
  (genuinely abstracting, not echoing). Caveat: all 144 eval calls were in the LoRA's training
  rows, so that number is style fit, not generalization.
- Next: held-out train/eval split, then a `Qwen/Qwen3-4B-Instruct-2507` LoRA on Runpod.

## Layout

- `eval/` — frozen eval set + scorer: ROUGE, copy/duplicate rate, bootstrap CI, grounding.
  `python eval/score.py` (needs `rouge_score`).
- `finetune/` — dataset build, Qwen3 LoRA training, inference (ported from meeting-summarizer).

## Credentials

`.env` holds `SUPABASE_URL` / `SUPABASE_SECRET` / `ORG_ID` — gitignored. `SUPABASE_SECRET` is a
`sb_secret_` key; it 401s on browser-like User-Agents, so use curl/urllib with the `apikey` header.
