# Working notes

## How this user wants to be taught

- **Depth: "just enough to build it."** Recipes and failure modes over derivations.
  Theory earns its place only when it changes a decision they have to make. Do not
  walk Hinton's temperature math; do explain why a different tokenizer kills logit
  matching, because that one decides the whole method.
- **Ground everything in their actual repo.** They respond to claims checked against
  their own files (`tokenizer.json`, `train.jsonl`, `results.jsonl`) far better than
  to generic statements. Prefer "I read your tokenizer, it is 151,669" over "Qwen3
  uses roughly 150k tokens."
- **Interactive verification beats assertion.** When a lesson makes a factual claim,
  give them a snippet that checks it on their machine. Automatic feedback, tight loop.

## Context worth remembering

- They invoke `teach` deliberately, so they want durable understanding, not a
  one-off script they will not be able to re-run.
- They picked **all four** motivations for going smaller (offline, cheap, fast,
  learning the technique). The learning goal is not decoration — a working model
  alone does not satisfy this mission.
- They are comfortable with TRL / PEFT / LoRA already, and have debugged a
  non-trivial chat-template problem unaided. Start lessons above that floor.

## Style

- Open every lesson file in the browser when it is written
  (`Start-Process <file>`), per the skill.
- Lessons are local files opened from disk, not published artifacts. The Artifact
  tool is unavailable in this session anyway (`ANTHROPIC_AUTH_TOKEN` takes
  precedence over a claude.ai login), so author full standalone HTML documents.
- **Write `data-correct="true"`, never bare `data-correct`.** A bare boolean
  attribute was silently stripped when lesson 06 was written with the Write tool —
  the file on disk had no `data-correct` at all, so every quiz question would have
  had *no* correct answer while looking perfectly normal in the source I sent.
  `assets/quiz.js` tests `opt.hasAttribute('data-correct')`, so any value works.
  After writing any quiz, verify with
  `grep -c data-correct lessons/NNNN-*.html` and expect exactly one per question.
  (Lessons 01–05 predate this and still carry bare attributes, which are fine as
  they stand — do not "fix" them into a mixed style.)

## Standing pre-checks before writing an explanation

Earned the hard way; both of these have already produced a wrong claim that reached
published documents.

- **Before attributing any latency outlier, check whether the machine slept.** An
  application timer cannot see a suspend, so the cause it suggests is a guess about a
  layer it cannot observe.
  ```powershell
  Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=506,507} |
    Select-Object -First 40 TimeCreated, Id
  ```
  IDs 506/507 are Modern Standby enter/exit. A suspend inside the run window explains a
  stall that no resource metric accounts for. See
  [learning-records/0003-the-machine-slept.md](learning-records/0003-the-machine-slept.md).
  **Confirmed twice, 2026-10-13.** The f16 run's stalls and the greedy run's single 8,088 s
  row both resolve to logged suspend windows (1,936 s and **8,065 s**, entered 10:03:36 /
  exited 12:18:01, against that row's 8,080.5 s prefill). The second time the query ran
  *before* any explanation was written, so nothing had to be retracted. Run it first; it
  costs seconds.
- **Read every outlier, not just the biggest.** If a proposed cause cannot produce all
  of them, it is not the cause. Two outliers with different signatures (prefill-inflated
  vs decode-inflated) still point at one mechanism if that mechanism can land in either
  phase.
- **Convert any reported duration into implied throughput** before trusting it. A field
  is a measurement only if the hardware could physically deliver it.
