# FlashMini — Status

Living status page. Updated at each milestone; claims here are backed by the
commands shown, not by intent.

## What works today

The deterministic tool layer, the local-model runtime, and the agent loop that drives them.

| Phase | Component | State |
|-------|-----------|-------|
| 1.1 | Environment foundation (uv, pytest, ruff, pyright, CI) | Done — CI green |
| 1.2 | Tool layer: 8 tools, normalized envelope, workspace jail, atomic patching | Done — 64 tests, smoke 10/10 |
| 1.3 | Model benchmark (Qwen3.5-4B vs Nanbeige4.2-3B vs Nemotron-Bolt) | **Done — winner: Qwen3.5-4B MTP, 4.2 tok/s, all gates pass** (`docs/decisions.md`) |
| 2 | Agent loop + CLI | In progress — runtime + agent loop + CLI landed; AgentBench v0 rounds: 7/8 → **8/8 → 8/8**; live smoke passes |

The agent loop now exists and drives the tool layer end-to-end: the model
must answer with exactly one JSON object per turn (a tool call or a final
answer), constrained server-side by `json_schema`; tool results return as
normalized envelopes; step budgets are enforced. Verified live: a full
read-and-report task passes (`scripts/smoke_agent.py`). The runtime layer
manages a local `llama-server` (health check, thinking-off chat,
grammar-constrained tool calls, prompt-cache reuse); the tool layer below
remains plain, deterministic Python.

## The tool layer (Phase 1.2)

Every tool returns the same envelope:

```json
{"tool": "...", "status": "ok|error|timeout|denied", "output": "...",
 "stderr": "...", "duration_ms": 0, "truncated": false, "next_action_hint": null}
```

Eight tools: `read_file`, `write_file`, `apply_patch`, `grep`,
`symbol_search`, `git_ops`, `run_tests`, `lint`.

Properties that are tested, not assumed:

- **Workspace jail** — path escapes (`../`, absolute paths, symlink tricks)
  come back as `denied`; writes inside `.git/` are refused.
- **Atomic patching** — Codex-style patch format, all-or-nothing: if any
  hunk fails to match, no file is touched.
- **Safe git** — strict allowlist (`status`, `diff`, `log`, `show`, `add`,
  `commit`, `rev-parse`, `branch`); destructive verbs are denied.
- **Contained failures** — the registry converts every exception into a
  clean `error` result; callers never see a stack trace.
- **Backups** — overwritten or deleted files are copied to
  `.flashmini/backups/<timestamp>/` first.

## Verify it yourself

```sh
make check        # ruff + format check + pyright + 96 tests
make smoke        # dogfoods all 8 tools against this repository
make smoke-agent  # live: the real model drives the agent loop (slow, minutes)
```

Both pass on the development machine (Lubuntu, i5-7300U, 8 GB RAM,
CPU-only).

## What's next

Phase 2 continues: benchmark and harden. AgentBench v0 (8 deterministic
tasks — read, search, patch, fix-test) scores the live agent end-to-end on
the winning model (Qwen3.5-4B MTP under `llama-server` with prompt caching
and `--spec-type draft-mtp --spec-draft-n-max 2`). Rounds: 7/8 → 8/8 → 8/8
— every weakness found gets fixed and re-measured, and the score repeats.
The CLI now ships the first workflows: `flashmini run "<task>"` and
`flashmini fix` (fix a failing test), with live step output.

The engineering plan for how a 4B local agent competes with models 100x
larger is in [`frontier-plan.md`](frontier-plan.md).

## Design documents

- [`DESIGN.md`](DESIGN.md) — architecture.
- [`VISION.md`](VISION.md) — why this project exists.
- [`frontier-plan.md`](frontier-plan.md) — how a 4B local agent competes with 100x larger models.
- [`decisions.md`](decisions.md) — dated decisions with measured evidence.
