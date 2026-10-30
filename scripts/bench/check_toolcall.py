"""Validate grammar-constrained tool-call output from a llama-cli run.

Looks only at the text generated after the final assistant marker, parses the
first complete JSON object there, and checks for the tool-call envelope
("tool" key plus an "args" object). Prompt echoes earlier in the file are
ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ASSISTANT_MARKER = "<|im_start|>assistant"


def find_toolcall_object(text: str) -> dict[str, object] | None:
    generated = text.rsplit(ASSISTANT_MARKER, 1)[-1]
    decoder = json.JSONDecoder()
    fallback: dict[str, object] | None = None
    for index, char in enumerate(generated):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(generated, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "tool" in obj:
            return obj
        if fallback is None:
            fallback = obj
    return fallback


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_toolcall.py <output-file>")
        return 2
    text = Path(argv[1]).read_text(encoding="utf-8", errors="replace")
    obj = find_toolcall_object(text)
    if obj is None:
        print("FAIL: no complete JSON object found (truncated or non-JSON output)")
        return 1
    ok = "tool" in obj and isinstance(obj.get("args"), dict)
    print("PARSED:", json.dumps(obj, ensure_ascii=False)[:300])
    print("HAS tool+args:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
