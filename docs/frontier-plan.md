# The Frontier Plan

How FlashMini — a 4B model on an 8 GB CPU laptop — is engineered to be
comparable to frontier systems like DeepSeek-V4.1-Flash-1017 on scoped,
verifiable work.

## The thesis, stated carefully

DeepSeek-V4.1-Flash-1017 is a frontier-class agentic model. Its published
model-card benchmarks — Terminal-Bench 4.0 42.3, DeepSWE v1.1 77.0,
NL2Repo-Bench 68.2, Automation-Bench 60.5 — are out of reach for any local
quantization: even its most extreme 1-bit-class quant weighs roughly 230 GB.
Any claim that a laptop 4B model matches it in general conversation is false,
and this project will not make that claim.

What is achievable — and what this plan is about — is **task-level
comparability where correctness is decidable**: fix this failing test, apply
this multi-file patch, answer this question about the repository, emit this
tool call in this protocol. In those regimes a small model can borrow the
integrity of its tools. The research supports exactly this shape:

- **T1 (ICLR 2026, arXiv:2504.04718)** — tool-integrated verification: an
  external filter (code interpreter) narrows candidates, then the small model
  verifies. A 1B model with T1 beats an 8B model. The theory: offloading
  checks to tools removes the memorization burden that small models fail.
- **Compiling Agentic Workflows into LLM Weights (arXiv:2605.22502)** — a
  fine-tuned 8B reaches 87–98% of in-context frontier quality on procedural
  tasks at 128–462× lower per-conversation cost.
- **FINEVERIFY (EMNLP 2026)** — fine-grained self-verification converts extra
  test-time compute into accuracy more reliably than voting; verification
  traces make failures auditable.

Discipline beats scale — when the discipline is real. The rest of this
document is the discipline, pillar by pillar, with receipts from this machine
(i5-7300U, 2 cores / 4 threads, no GPU, 7.6 GiB RAM) where they exist.

## 1. Architecture — the loop is the product

The agent is a state machine, not a chat window: classify → plan → retrieve →
act → verify → repair → finalize → escalate. Two speeds: **Flash** (single
pass, fast iteration) and **Deep** (bounded repair, up to five iterations,
plus candidate filtering when the task warrants it).

Mechanisms that carry the quality:

- **Verification gates.** Nothing is "fixed" until tests and linters pass.
  The claim and the evidence are the same object. This is the T1 result made
  operational: the model proposes, the tools decide.
- **Tool-first reasoning.** Read before write; plans are expressed as tool
  calls, so a plan is checkable before it is executed.
- **Candidate → filter → verify.** For ambiguous fixes: generate a small
  number of attempts, filter with tests/lint, verify survivors with a
  fine-grained pass. Once telemetry exists, budget allocation follows
  SPLIT (arXiv:2610.23255) instead of fixed rules.
- **Bounded repair.** Deep mode repairs at most five times, each driven by a
  structured error envelope (`next_action_hint`), then escalates honestly.
  Never infinite retry; never pretend.
- **Escalation ladder.** Retry → Deep mode → report failure with evidence.
  Dev-side teacher models may later convert solved failures into training
  data. They are never runtime.

Status: designed. Phase 2 builds the Flash loop; Deep mode and guarded
candidate filtering arrive in Phase 4.

## 2. Tokenizer and protocol — zero-parse-failure by construction

- **Native llama.cpp tokenization** — no re-tokenization drift between the
  agent's accounting and the model's reality.
- **GBNF grammars wherever structure matters.** Tool calls are grammar-
  constrained and measured at 100% compliance in both benchmark rounds
  (bench script: `scripts/bench/toolcall_test.sh`). The grammar suite extends
  to the patch envelope and plan objects. A model that cannot emit an invalid
  tool call has failures of judgment — never of syntax.
- **Token economics.** The 248K vocabulary packs code into fewer tokens than
  older 32K vocabularies — every file costs less context and less time. Per-
  task-class budgets; tool outputs truncated with explicit `truncated` flags;
  compaction policy runs before context overflow, not after.

## 3. Weights — small on purpose, measured on this machine

| candidate | quant | size | gen t/s | verdict |
|---|---|---|---|---|
| Nanbeige4.2-3B | Q4_K_M | 2.50 GiB | 2.73 | fails speed gate (looped transformer: 44 effective passes) |
| Qwen3.5-4B | UD-Q4_K_XL | 2.71 GiB | 3.61 | close; converges with MTP |
| **Qwen3.5-4B MTP** | **UD-Q4_K_XL** | **2.78 GiB** | **4.2** | **winner** — all gates pass |

- The winner: **Qwen3.5-4B with multi-token-prediction heads**, hybrid Gated
  DeltaNet + attention, vision encoder included, apache-2.0. Config:
  `--spec-type draft-mtp --spec-draft-n-max 2` (draft depth is tuned, not
  assumed: n-max 2 beats 3 and 6 on this CPU).
- **Quantization is a compute lever, not only a memory lever.** CPU
  generation is memory-bandwidth bound (thread sweep is flat: 3.66 / 3.61 /
  3.57 t/s at 2 / 3 / 4 threads), so bits convert directly into speed.
  Mid-bit quants preserve capability (arXiv:2601.14277); UD-Q4_K_XL is the
  quality/speed equilibrium.
- **MTP heads are trained-in drafting** — no separate draft model file, no
  second model in RAM.
- Path to more capability at the same weight: the **distillation flywheel**
  (verified traces only → SFT on Kaggle T4×2 → "FlashMini Ultra"), and for
  stable procedures, compilation-style training where the workflow lives in
  the weights and the prompt stays constant-size.

## 4. Computation — seconds are a feature

The agent's usability is bound by latency, so each llama.cpp lever is
measured, not assumed.

Measured so far:

- MTP `n-max 2`: **4.2 t/s** vs 3.3 control on the same file (+27%).
- `ngram-cache`: 2.8 t/s — needs a larger history than this workload has.
- Over-drafting hurts: n-max 6 collapses to 2.5 t/s. Community consensus
  matches: speculative decoding below ~40% acceptance is not worth the
  overhead.

Planned levers, in impact order:

1. **Prompt caching (Phase 2, the big one).** An agent iteration re-sends
   system prompt + repository context + history + the newest tool result.
   At 12–17 t/s prompt processing, recomputing that prefix would dominate
   every iteration. `llama-server`'s slot cache keeps the shared prefix hot;
   iteration cost then scales with the delta, not the transcript.
2. **KV cache quantization** (`-ctk` / `-ctv`): measured on this machine —
   it costs generation speed (q8_0 −15%, q4_0 −23% vs f16), so f16 stays the
   default; kept as the memory lever of record if context grows. Sub-4-bit KV
   (TurboQuant-class) stays on the watchlist, not in production.
3. Batching (`-b 512` / `-ub 512`), flash attention where applicable,
   draft-KV quantization (`-ctkd` / `-ctvd`), thread placement (`-t` / `-tb`).
4. Workload-specific speculators: the upstream decision matrix maps code
   refactoring to `ngram-simple`, structured output to `ngram-map-k`, long
   sessions to `ngram-mod`. Re-test per workload class.
5. Later: self-speculative scaling (QuantSpec-class, >90% acceptance in
   research) if it lands in mainline.

## 5. Solutions — the unglamorous correctness layer

Done (Phase 1.2, committed, 64 tests, `make smoke` dogfooding all eight
tools against this repository):

- Eight deterministic tools behind one normalized envelope, including
  `next_action_hint` for repair loops.
- Workspace jail (path/symlink escapes → `denied`), atomic all-or-nothing
  patch application, backups to `.flashmini/backups/`, git allowlist with
  destructive verbs denied, test/lint gates.

Planned:

- **Code intelligence (Phase 3):** tree-sitter symbol index and a PageRank-
  style repo map (the Aider design, rebuilt against our tool layer), plus
  hybrid retrieval — BM25 + sqlite-vec embeddings + reciprocal rank fusion,
  incremental.
- **Memory (Phase 4):** episodic / semantic / procedural / workspace stores
  in SQLite, with secret redaction.
- **Observability:** JSONL traces for every step; `flashmini trace <id>`
  replays them. `docs/decisions.md` is the method: every claim gets a date,
  a number, and a source.

## 6. The proof — comparability or it didn't happen

Scoped suites, built as the phases land:

1. **Tool-call protocol conformance** — deterministic, already 100% on the
   benchmark harness.
2. **Bugfix-with-tests** — failing test to passing, no regressions, bounded
   transcript.
3. **Repo QA** — questions answered with correct file/line citations.
4. **Multi-file patch tasks** — described change applied across files,
   verified by tests.

Every run records pass rate, tokens in/out, wall-clock, and peak RSS. A
frontier baseline column is produced dev-side (NVIDIA NIM API) on the same
suites — the mission's north star. Results are published in this repository,
whether or not they flatter the project.

Eval hygiene follows the field's hard lessons: binary checks before
model-graded ones; rubrics calibrated or not trusted; saturation watched
(an eval at 100% stops measuring); transcripts kept so failures can be
audited rather than argued about.

"Comparable" here means: **matched pass rates on these scoped suites at
usable latency on this hardware** — not matched open-ended conversation.

## Roadmap

| phase | delivers | serves |
|---|---|---|
| 2 | Agent loop + CLI: llama-server with prompt caching, grammar tool calls, Flash mode, first workflow (fix a failing test) | 1, 2, 4 |
| 3 | Code intelligence: tree-sitter symbols, repo map, hybrid retrieval | 5 |
| 4 | Memory + self-correction: Deep mode, guarded candidate filtering, repair telemetry | 1, 5 |
| 5 | Observability polish + first published benchmark table vs frontier baseline | 6 |
| 6 | FlashMini Ultra: distillation from verified traces (NIM teacher dev-side, Kaggle GPUs) | 3 |

## Honest limits

- **Bandwidth floor.** ~270 ms per generated token at the best measured
  config. Streaming and parallel retrieval/verification hide part of it;
  nothing hides all of it.
- **Model ceiling.** Long-horizon planning on open-ended tasks will stay
  weaker than frontier models. The plan routes work into bounded, verifiable
  scopes and escalates honestly when a task is not that shape.
- **Resource discipline.** One large model in RAM at a time; the measured
  peak footprint of the chosen configuration is ~4.5 GiB on a 7.6 GiB
  machine.
- **Claims discipline.** Every "planned" item above becomes real only when
  its numbers appear in `docs/decisions.md`. Until then it is a design, and
  it is labeled as one.
