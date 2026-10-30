#!/usr/bin/env bash
# Vendored binaries for FlashMini. Pinned versions, SHA256-verified, no sudo.
# Usage: bash scripts/fetch_bins.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor/bin"
mkdir -p "$VENDOR"

RG_VERSION="15.2.0"
RG_SHA256="33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c"
RG_ASSET="ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl"
RG_URL="https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/${RG_ASSET}.tar.gz"

if [ -x "$VENDOR/rg" ] && "$VENDOR/rg" --version 2>/dev/null | head -1 | grep -q " $RG_VERSION"; then
    echo "rg $RG_VERSION already vendored at vendor/bin/rg"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading ripgrep $RG_VERSION ..."
curl -fsSL "$RG_URL" -o "$tmp/rg.tar.gz"

echo "verifying sha256 ..."
echo "$RG_SHA256  $tmp/rg.tar.gz" | sha256sum -c -

tar -xzf "$tmp/rg.tar.gz" -C "$tmp"
install -m 0755 "$tmp/$RG_ASSET/rg" "$VENDOR/rg"
"$VENDOR/rg" --version | head -1
echo "installed: vendor/bin/rg"
