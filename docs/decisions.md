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
