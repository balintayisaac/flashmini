# FlashMini design

FlashMini is a local-first, terminal AI coding assistant that runs a small quantized model on CPU-only, 8 GB-class Linux hardware. This document is the condensed architecture reference. The project targets honest, verifiable behavior over impressive demos.

## Principles

1. **Local-first.** No cloud APIs by default. Everything runs on-device.
2. **Verification over claims.** The agent may not report "fixed" unless the relevant verification (tests, linters, build) passes.
3. **Strict tool results.** Every tool call returns normalized JSON: `{tool, status, output, stderr, duration_ms, truncated, next_action_hint}`. Tool outputs are never fabricated.
4. **Bounded resources.** One small model resident at a time; hard memory budget; native tokenizers only.
5. **Observable.** JSONL run logs with per-decision tracing, replayable via `flashmini trace <id>`.

## Agent core

The agent is an explicit state machine, not an unconstrained chat loop:

```
classify -> plan -> retrieve -> act -> verify -> repair -> finalize
                                        ^           |
                                        +-----------+ (bounded)
```

- **classify**: route the task into a class (question, single-file edit, multi-file change, bugfix, refactor, repo QA, deep debug).
- **plan**: produce a short, inspectable plan; task class bounds plan depth.
- **retrieve**: pull only relevant context (repo map, symbols, hybrid search hits, memory).
- **act**: execute tool calls.
- **verify**: run tests/linters/build checks appropriate to the task class.
- **repair**: on verification failure, diagnose and retry with a bounded iteration count (Flash: small; Deep: max 5).
- **finalize**: summarize with evidence; escalate to the user when the repair budget is exhausted or confidence is low.

Two reasoning modes share the same machinery: **Flash** (fast, minimal retrieval, no extended repair) and **Deep** (retrieval-heavy, bounded multi-iteration repair).

## Model runtime

- Small quantized GGUF LLM in the 1.5B–4B class (e.g. Qwen2.5-Coder variants) via llama.cpp, CPU-only with AVX2.
- Strict JSON-schema tool calling; malformed calls are normalized or rejected, never guessed.
- One model resident at a time. Larger models load only on explicit request with a memory check.

## Tool layer

File read/write, patch application, grep, symbol search, git operations, test runner, and linter invocation. Each tool: timeout, sandboxed execution, confirmation gate for destructive operations, and the normalized JSON result envelope. `next_action_hint` helps the agent decide its next state transition without extra round-trips.

## Code intelligence

- **tree-sitter**: AST-based chunking so retrieval units respect syntax boundaries.
- **universal-ctags**: symbol tables for navigation and repo maps.
- **ripgrep**: fast textual search backing grep tools and repo map generation.

## Retrieval

Hybrid search over the codebase:

- **Lexical**: SQLite FTS5/BM25.
- **Dense**: small local embeddings (bge-small class) stored in sqlite-vec.
- **Fusion**: reciprocal-rank fusion of both channels.
- **Indexing**: incremental; re-indexes only changed files, driven by mtime/hash checks.

## Memory

SQLite-backed stores with secret redaction applied before any write:

- **episodic**: past sessions and outcomes.
- **semantic**: distilled facts about the codebase and user preferences.
- **procedural**: successful workflows and repair strategies.
- **workspace**: current project state.

User-facing commands: `flashmini memory remember ...` / `flashmini memory forget ...`.

## Self-correction

Errors, failing tests, lint output, and build failures feed a bounded repair loop. The repair policy is part of the agent state machine, with per-task-class budgets. A run that exhausts its budget ends in an explicit escalation, never a false success report.

## Multimodal (optional, on-demand)

A small vision-language model loads only when an image is supplied; Tesseract OCR is the lightweight fallback for text extraction. Neither is resident by default.

## Security

- Sandboxed execution of generated commands (bubblewrap/firejail where available), with timeouts everywhere.
- Confirmation for dangerous operations (deletions, force-pushes, anything outside the workspace).
- Secret redaction in all persisted state: memory, logs, and trace files.

## CLI surface

Task commands: `ask`, `fix`, `edit`, `test`, `explain`, `repo-map`, `memory`, `image`, `deep`, `flash`.
Setup and maintenance: `install`, `download-models`, `index`, `prune-cache`, `status`, `doctor`, `benchmark`, `clean`.
Observability: `flashmini trace <id>`.

## Implementation phases

1. Base environment (Python toolchain, native deps, llama.cpp build)
2. Model runtime (GGUF load, quantized CPU inference, tokenizers)
3. Tool layer (normalized JSON results, sandboxing, timeouts)
4. Code intelligence (tree-sitter, ctags, ripgrep maps)
5. Retrieval (hybrid index, incremental updates)
6. Agent loop (state machine, Flash/Deep modes)
7. Memory (SQLite stores, redaction, remember/forget)
8. Self-correction (verification-driven repair)
9. Multimodal (optional VLM/OCR)
10. Polish (full CLI, doctor/benchmark, packaging)

Each phase ends with tests green before the next begins.
