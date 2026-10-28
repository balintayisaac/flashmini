# FlashMini

<p align="center">
  <img src="assets/logo.png" alt="FlashMini logo" width="140">
</p>

A terminal-first AI coding assistant for Linux that runs entirely on your machine: no cloud APIs, no account required, no telemetry. Built for the hardware most AI tooling ignores — 8 GB of RAM, four CPU cores, integrated graphics.

> **Discipline beats scale.** Wrap a small model in verification, retrieval, and memory rigorous enough, and it stops mattering that the cloud assistant has a hundred times the parameters. See [docs/VISION.md](docs/VISION.md).

**Status: active development.** The architecture is designed and documented in [docs/DESIGN.md](docs/DESIGN.md); implementation is underway. Capabilities described below are design targets, not shipped features. Nothing here should be treated as working until a release notes it as verified.

## Why

Local coding assistants tend to assume either a datacenter or a 24 GB workstation. Meanwhile plenty of perfectly capable laptops carry 8 GB of RAM and a 4-core CPU. FlashMini is designed around that constraint from the first line: a small quantized model, strict memory budgeting, and an agent loop that spends tokens on verification rather than verbosity.

## Design targets

- **Model:** small quantized GGUF LLM (1.5B–4B class, e.g. Qwen2.5-Coder family) served through llama.cpp, CPU-only, with a hard resident-memory budget.
- **Agent core:** an explicit state machine — classify, plan, retrieve, act, verify, repair, finalize, escalate. Two reasoning modes: *Flash* for quick tasks, *Deep* for multi-step work with bounded repair iterations.
- **Tool layer:** file read/write, patch application, grep, symbol search, git, test runner, linter — each call normalized into a strict JSON result with status, duration, and stderr. No fabricated tool output, ever.
- **Code intelligence:** tree-sitter AST chunking, universal-ctags symbol maps, ripgrep-backed repo maps.
- **Retrieval:** hybrid search (SQLite FTS5/BM25 + vector embeddings with reciprocal-rank fusion), incremental indexing that survives incremental edits.
- **Memory:** episodic, semantic, procedural, and workspace stores in SQLite, with secret redaction before anything is written.
- **Self-correction:** errors, failing tests, and lint output drive a bounded repair loop. The assistant never claims "fixed" without the verification command passing.
- **Observability:** JSONL run logs with per-decision tracing, replayable via `flashmini trace <id>`.
- **Security:** sandboxed tool execution, timeouts, explicit confirmation for destructive operations, secret redaction in all persisted state.

## Hardware targets

| | Minimum | This project's dev machine |
|---|---|---|
| CPU | x86-64 with AVX2 | Intel i5-7300U (4 threads) |
| RAM | 8 GB | 8 GB |
| Disk | ~5 GB free (model + index) | 48 GiB |
| GPU | none required | none |

## Roadmap

1. Base environment: Python tooling, native build deps, llama.cpp runtime
2. Model runtime: GGUF loading, quantized inference, tokenizer integration
3. Tool layer: file/git/test/linter tools with normalized JSON results
4. Code intelligence: tree-sitter, ctags, ripgrep repo maps
5. Retrieval: hybrid BM25 + vector search, incremental indexing
6. Agent loop: state machine, Flash/Deep modes
7. Memory: SQLite stores, remember/forget, redaction
8. Self-correction: verification-driven repair loop
9. Multimodal (optional): on-demand small VLM, OCR fallback
10. Polish: CLI surface, doctor/benchmark commands, packaging

## License

MIT — see [LICENSE](LICENSE).
