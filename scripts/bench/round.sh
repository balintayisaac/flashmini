#!/usr/bin/env bash
# Crash-safe AgentBench round driver.
#
# Runs the bench detached from any calling terminal (setsid + nohup), captures
# every byte of output to a persistent log under bench/results/, kills any
# orphaned FlashMini llama-server first, and prints the final score.
#
# Usage:
#   scripts/bench/round.sh [--label LABEL] [agentbench args...]
#   COMMIT=1 scripts/bench/round.sh ...   # also commit+push results on a clean pass
#
# Launch fully detached (survives harness/terminal loss):
#   setsid nohup scripts/bench/round.sh --label full --verbose \
#       </dev/null >/tmp/opencode/round-driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

label=""
if [ "${1:-}" = "--label" ]; then
  label="$2"
  shift 2
fi
ts=$(date -u +%Y%m%d-%H%M%S)
log="bench/results/round-${ts}${label:+-$label}.log"

# Only our own vendor server is ours to kill (path-scoped; never ollama's).
pkill -f 'FlashMini/vendor/llama/llama-server' 2>/dev/null && sleep 2

echo "bench log: $log"
uv run python scripts/bench/agentbench.py "$@" >"$log" 2>&1
rc=$?
tail -4 "$log"

json=$(grep -oE 'bench/results/agentbench-[0-9]+-[0-9]+\.json' "$log" | tail -1)
if [ -n "$json" ] && [ -f "$json" ]; then
  python3 - "$json" <<'PY'
import json, sys

d = json.load(open(sys.argv[1]))
per_task = ", ".join(
    f"{t['name']}={'PASS' if t['passed'] else 'FAIL'}" for t in d.get("tasks", [])
)
print(f"SCORE {d.get('passed')}/{d.get('total')}  [{per_task}]")
PY
else
  echo "NO RESULTS JSON — bench crashed; full log: $log"
fi

if [ "${COMMIT:-0}" = "1" ] && [ "$rc" -eq 0 ] && [ -n "${json:-}" ]; then
  git add "$json"
  git commit -q -m "AgentBench round (${label:-full}): all tasks pass" &&
    git push -q origin main 2>/dev/null &&
    echo "committed+pushed"
fi
exit "$rc"
