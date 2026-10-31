# FlashMini decision log

Dated decisions with the evidence behind them. Measured numbers come from this
machine (i5-7300U, 2 cores / 4 threads, no GPU, 7.6 GiB RAM) unless noted.

## 2026-10-30 — Phase 1.3 benchmark setup

Runtime: llama.cpp nightly **b12001** (`0.8.0-dev`, commit eb4e244be), prebuilt
ubuntu-x64 CPU binaries vendored at `vendor/llama/` (gitignored). Loaded CPU
backend: `libggml-cpu-haswell.so`.

Benchmark gates (from the plan): generation speed ≥ 4 tok/s, peak RSS ≤ 5.5 GiB
at usable context, 100% JSON tool-call compliance under a GBNF grammar.

Tooling: `scripts/bench/toolcall.gbnf` (JSON tool-call envelope grammar),
`scripts/bench/toolcall_test.sh` (grammar-constrained compliance run),
`scripts/bench/check_toolcall.py` (output validator).

## Candidate: Nanbeige4.2-3B Q4_K_M

File: `bartowski/Nanbeige_Nanbeige4.2-3B-GGUF`, `Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf`,
2.50 GiB, 4.17 B params, looped transformer (22 layers × 2 loops = 44 passes).

| test | result | gate | verdict |
|---|---|---|---|
| pp512 (prompt processing) | 10.02 tok/s | — | very slow |
| tg128 (generation) | 2.73 tok/s | ≥ 4 | **fail** |
| peak RSS, `-c 4096` | 4.36 GiB | ≤ 5.5 GiB | pass |
| peak RSS, default ctx (unset `-c`) | 6.42 GiB | ≤ 5.5 GiB | n/a — always set `-c` explicitly |
| coherence sample (Python function + docstring) | correct and well structured; enters thinking mode by default | — | pass |
| JSON tool-call compliance (GBNF) | 2/2 runs emitted `{"tool": "read_file", "args": {...}}` under the grammar | 100% | pass |

Notes: the looped architecture doubles effective compute at equal parameter
count, which matches the roughly 2× slower generation versus what a normal 3 B
dense model should achieve here. A DSpark draft model (556 MiB) exists for
speculative decoding as a possible later speed-up.

Status: speed gate failed at 2.73 tok/s; comparison candidate Qwen3.5-4B
(next entry) will be benchmarked on the same protocol before a verdict.

## Candidate: Qwen3.5-4B UD-Q4_K_XL

File: `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-UD-Q4_K_XL.gguf`, 2.71 GiB, 4.21 B
params, hybrid Gated DeltaNet + Gated Attention (32 layers).

| test | result | gate | verdict |
|---|---|---|---|
| pp512 (prompt processing) | 17.17 tok/s | — | slow, as expected on CPU |
| tg128 (generation, `-t 4`) | 3.61 tok/s | ≥ 4 | **fail (close)** |
| thread sweep (`-t 2/3/4`) | 3.66 / 3.61 / 3.57 tok/s | — | flat — memory-bandwidth bound |
| peak RSS, `-c 4096` | 3.86 GiB | ≤ 5.5 GiB | pass |
| coherence sample | correct Python function + docstring; thinking mode on by default | — | pass |
| JSON tool-call compliance (GBNF) | emitted `{"tool": "read_file", "args": {"path": "pyproject.toml"}}` with the correct argument name | 100% | pass |

Notes: baseline sits within ~10% of the speed gate. Next lever: the MTP
variant (`unsloth/Qwen3.5-4B-MTP-GGUF`, same quants plus multi-token-prediction
heads), run with `--spec-type draft-mtp --spec-draft-n-max 3`. Community
numbers for Qwen MTP on other hardware: 1.5–1.7x speedup at 70–85% draft
acceptance; highest gains on structured output such as code and tool calls,
which is exactly FlashMini's workload.

## Verdict: Qwen3.5-4B MTP UD-Q4_K_XL — winner (2026-10-31)

File: `unsloth/Qwen3.5-4B-MTP-GGUF`, `Qwen3.5-4B-UD-Q4_K_XL.gguf`, 2.78 GiB
(2,990,664,000 bytes) — the base weights plus multi-token-prediction heads
(`mtp_shared_head_norm`, `mtp_h_input`, `mtp_tok_embd` tensors confirmed in
the compute graph; 441 tensors). Engagement verified in verbose logs:
`common_speculative_init_result: creating MTP draft context against the
target model`.

| spec configuration | generation | prompt | peak RSS | verdict |
|---|---|---|---|---|
| `--spec-type none` (control) | 3.3 t/s | 12.2 t/s | — | baseline |
| `--spec-type draft-mtp --spec-draft-n-max 6` | 2.5 t/s | 10.8 t/s | 4.68 GiB | fails — over-drafting |
| `--spec-type ngram-cache` | 2.8 t/s | 13.0 t/s | — | fails at this corpus size |
| `--spec-type draft-mtp --spec-draft-n-max 3` | 3.7 t/s | 10.7 t/s | 4.53 GiB | small gain |
| **`--spec-type draft-mtp --spec-draft-n-max 2`** | **4.2 t/s** | 12.1 t/s | ~4.5 GiB | **PASS — best** |

All gates pass with the winning config: generation 4.2 tok/s (≥ 4), peak RSS
~4.5 GiB at `-c 4096` (≤ 5.5), grammar-constrained tool-call compliance 100%
(correct `path` argument), coherence good. Draft length matters more than
expected on a 2-core CPU: n-max 2 beats 3 and 6 — deeper drafts are wasted
work at moderate acceptance. ngram-cache needs a larger history to pay off.
Nemotron-Bolt round skipped: weaker agent/tool benchmarks in its class and
larger file; noted as re-checkable if a specific failure mode appears.

## Runtime API validation — llama-server b12001 (2026-10-31)

Probed live against the vendored server with the winning config
(`-t 4 -c 4096 --spec-type draft-mtp --spec-draft-n-max 2`):

| check | result |
|---|---|
| `/health` | `{"status":"ok"}` once loaded |
| chat endpoint | OpenAI-compatible `/v1/chat/completions` |
| thinking control | `chat_template_kwargs: {"enable_thinking": false}` required; otherwise `content` is empty and tokens go to `reasoning_content` |
| `cache_prompt` | prefix reuse confirmed (`cached_tokens: 11/15`; the second request evaluated only the 4-token delta) |
| MTP acceptance (logged) | 1.000 (4/4) on short calls; 0.778 (77/99) on a longer call |
| per-request `json_schema` | enforced server-side; produced a valid tool-call object `{"tool": "read_file", "args": {"path": "pyproject.toml"}}` |

The runtime layer (`src/flashmini/runtime/`) therefore uses thinking-off by
default, `json_schema` for tool calls, and `cache_prompt: true` on every
request.

## AgentBench v0 rounds 1–2 — 7/8 → 8/8 (2026-10-31)

Harness: `scripts/bench/agentbench.py` — 8 deterministic tasks (read and
report, symbol search, text search, fix-a-failing-test, patch-update, plus
three tool-usage protocol checks). Runs the live agent loop on the winning
model in disposable workspaces; results JSONs are committed under
`bench/results/`.

| round | score | notes |
|---|---|---|
| 1 | 7/8 (88%) | `patch-update` failed: the model emitted unified-diff syntax seven times against the `*** Begin Patch` parser (164 s, hit the step cap; file untouched — atomicity held) |
| diagnostic | 0/1 | `--only patch-update --verbose` captured every raw reply and error (tooling added this round: `--only`, `--verbose`, per-step records in the results JSON) |
| 2 | **8/8 (100%)** | after the fixes below; `patch-update` passed in 59 s via `read_file` → `apply_patch` → `git_ops` |

Fixes between rounds — each one a product gap the score exposed:

1. `apply_patch` errors now teach the expected format: the error output
   always carries the template, and unified-diff input gets a specific
   "no `--- a/` / `+++ b/` header lines" explanation.
2. The system prompt shows the patch template up front and adds the rule
   "if a tool call fails, do not repeat it unchanged".
3. The loop warns the model when an identical call has already failed
   ("change the arguments or the approach") instead of letting it retry
   blindly.

Round 2 per-task wall time: 16–97 s (fastest: the symbol/grep protocol
checks; slowest: `fix-test` at 96.8 s with two pytest runs and a patch).
The frontier/other-model comparison column in this table is produced via
`--base-url` against an OpenAI-compatible endpoint (dev-side; pending key).
