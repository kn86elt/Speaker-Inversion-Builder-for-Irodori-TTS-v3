#!/usr/bin/env python3
"""Transcribe WAV files using faster-whisper in the WebUI environment."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_paths", nargs="+")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run `uv sync --no-dev` in the WebUI project."
        ) from exc

    compute = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model_size, device=args.device, compute_type=compute)
    for wav_path in args.wav_paths:
        segments, _ = model.transcribe(wav_path, language=args.language, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        print(
            json.dumps({"path": wav_path, "text": text, "device": args.device}, ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
