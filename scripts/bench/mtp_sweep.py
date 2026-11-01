#!/usr/bin/env python3
"""A1: MTP draft-depth sweep (frontier-plan.md Track A1).

Starts the vendored llama-server on the production model with
--spec-draft-n-max swept over candidate depths, measures generation
tok/s (median of 3 completions after 1 warmup) and peak RSS (VmHWM),
then writes a results JSON under bench/results/.

Guards: refuses to run while a bench round/probe or another llama-server
is up (one-model-in-RAM rule). Baseline depth 0 omits the spec flags.

Usage: uv run python scripts/bench/mtp_sweep.py [--depths 0,1,2,3,4,6]
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "Qwen3.5-4B-MTP-UD-Q4_K_XL.gguf"
SERVER = REPO / "vendor" / "llama" / "llama-server"
PORT = 47901
BASE_ARGS = ["--host", "127.0.0.1", "--port", str(PORT), "-t", "4", "-c", "4096"]

PROMPT = (
    "Write a Python function that merges two sorted lists into one sorted "
    "list without using sort(). Include docstring, edge cases, and three "
    "doctests. Then explain the time complexity."
)
N_PREDICT = 128


def check_clear() -> None:
    """Abort if any bench run or llama-server is already active."""
    for pattern in ("agentbench.py", "round.sh", "llama-server"):
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        ).stdout.strip()
        if out:
            raise SystemExit(
                f"refusing to run: '{pattern}' already active (pids: {out}). "
                "One model in RAM at a time."
            )


def start_server(depth: int, cache_type: str) -> subprocess.Popen:
    args = [str(SERVER), "-m", str(MODEL), *BASE_ARGS]
    if cache_type != "f16":
        args += ["-ctk", cache_type, "-ctv", cache_type]
    if depth > 0:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(depth)]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_healthy(proc: subprocess.Popen, timeout: float = 180.0) -> None:
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early (depth sweep), rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2.0)
    raise SystemExit("server did not become healthy in time")


def complete(n_predict: int = N_PREDICT) -> dict:
    payload = json.dumps(
        {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    return body.get("timings", {})


def peak_rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return -1


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    time.sleep(3.0)  # let the kernel reclaim before the next load


def measure_depth(depth: int, cache_type: str, warmup_done: bool) -> dict:
    proc = start_server(depth, cache_type)
    try:
        wait_healthy(proc)
        if not warmup_done:
            complete()  # warmup: page cache, allocator, first-token paths
        rates = []
        for _ in range(3):
            timings = complete()
            rate = timings.get("predicted_per_second")
            if rate is None:
                raise SystemExit(f"no predicted_per_second in timings: {timings}")
            rates.append(round(float(rate), 3))
        rss = peak_rss_kb(proc.pid)
        return {
            "depth": depth,
            "cache_type": cache_type,
            "tok_s_median": round(statistics.median(rates), 3),
            "tok_s_all": rates,
            "peak_rss_gib": round(rss / 1024 / 1024, 3),
        }
    finally:
        stop_server(proc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", default="0,1,2,3,4,6", help="comma list; 0 = no MTP")
    ap.add_argument(
        "--cache-types",
        default="f16",
        help="comma list of KV cache types to sweep (e.g. f16,q8_0)",
    )
    args = ap.parse_args()
    depths = [int(d) for d in args.depths.split(",")]
    cache_types = [c for c in args.cache_types.split(",") if c]

    check_clear()
    if not MODEL.exists():
        raise SystemExit(f"model not found: {MODEL}")

    results = []
    first = True
    for depth in depths:
        for cache_type in cache_types:
            label = "baseline (MTP off)" if depth == 0 else f"n_max={depth}"
            label += f" kv={cache_type}"
            print(f"[mtp_sweep] depth {label} ...", flush=True)
            row = measure_depth(depth, cache_type, warmup_done=first)
            first = False
            results.append(row)
            print(
                f"[mtp_sweep]   {row['tok_s_median']} tok/s "
                f"(runs {row['tok_s_all']}), RSS {row['peak_rss_gib']} GiB",
                flush=True,
            )

    out = {
        "tool": "mtp_sweep",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL.name,
        "server_args": BASE_ARGS,
        "cache_types": cache_types,
        "n_predict": N_PREDICT,
        "results": results,
    }
    dest = REPO / "bench" / "results" / f"mtp-sweep-{datetime.now():%Y%m%d-%H%M%S}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[mtp_sweep] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
