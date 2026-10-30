#!/usr/bin/env bash
# Grammar-constrained tool-call compliance test (Phase 1.3 bench).
# Usage: scripts/bench/toolcall_test.sh <model.gguf> [output-file]
# Uses a ChatML-style prompt (Nanbeige / Qwen3.5 families) and forces the
# output through scripts/bench/toolcall.gbnf so only JSON can be emitted.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
LLAMA="$REPO_ROOT/vendor/llama/llama-cli"
GRAMMAR="$BENCH_DIR/toolcall.gbnf"

MODEL="$1"
OUT="${2:-/tmp/opencode/toolcall_out.txt}"

"$LLAMA" -m "$MODEL" -t 4 -c 2048 -n 128 \
  --grammar-file "$GRAMMAR" --no-display-prompt \
  -p "<|im_start|>user
You are FlashMini's planner. Reply with exactly one tool call as compact JSON of the form {\"tool\": \"<tool_name>\", \"args\": {...}}. Task: read the file pyproject.toml.
<|im_end|>
<|im_start|>assistant
" > "$OUT" 2> "$OUT.err"

python3 "$BENCH_DIR/check_toolcall.py" "$OUT"
