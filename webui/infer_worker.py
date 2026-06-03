#!/usr/bin/env python3
"""Run Irodori infer.py from a UTF-8 JSON request.

This keeps generated text out of the Windows process command line so Japanese
text, whitespace, and newlines are passed to infer.py as Python strings.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: infer_worker.py <infer.py> <request.json>")

    script = Path(sys.argv[1]).resolve()
    request_path = Path(sys.argv[2]).resolve()
    with request_path.open(encoding="utf-8") as f:
        req = json.load(f)

    argv = [
        str(script),
        "--checkpoint", str(req["checkpoint"]),
        "--ref-embed", str(req["embedding_path"]),
        "--text", str(req["text"]),
        "--output-wav", str(req["output_wav"]),
        "--num-steps", str(req["num_steps"]),
    ]
    seed = int(req.get("seed", -1))
    if seed >= 0:
        argv += ["--seed", str(seed)]

    old_argv = sys.argv
    try:
        sys.path.insert(0, str(script.parent))
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
