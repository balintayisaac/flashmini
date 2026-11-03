"""Diagnose Kaggle egress: DNS + HTTPS reachability for the hosts Track B needs.

Runs in seconds on CPU. Prints a compact table; no training, no GPU.
"""

import socket
import subprocess
import time

HOSTS = [
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "github.com",
    "pypi.org",
    "integrate.api.nvidia.com",
    "www.kaggle.com",
]

print("[diag] socket.gethostbyname probe:")
for host in HOSTS:
    t0 = time.time()
    try:
        ip = socket.gethostbyname(host)
        print(f"  DNS OK   {host} -> {ip} ({time.time() - t0:.1f}s)")
    except Exception as exc:
        print(f"  DNS FAIL {host}: {type(exc).__name__}: {exc} ({time.time() - t0:.1f}s)")

print("[diag] HTTPS HEAD probe (curl, 15s timeout each):")
for host in HOSTS:
    r = subprocess.run(
        ["curl", "-sI", "-m", "15", "-o", "/dev/null", "-w", "%{http_code}", f"https://{host}"],
        capture_output=True,
        text=True,
    )
    print(f"  HTTPS {host}: status={r.stdout!r} rc={r.returncode}")

print("[diag] pip index reachability:")
r = subprocess.run(
    ["pip", "download", "--no-deps", "-d", "/tmp/pipdiag", "six"],
    capture_output=True,
    text=True,
    timeout=120,
)
_tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-120:]
print(f"  pip download six: rc={r.returncode} tail={_tail}")

print(
    "[diag] env proxies:",
    {k: v for k, v in __import__("os").environ.items() if "proxy" in k.lower() or "PROXY" in k},
)
