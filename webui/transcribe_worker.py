#!/usr/bin/env python3
"""Transcribe one WAV using the Irodori uv environment."""
from __future__ import annotations

import argparse
import json


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    device = "cuda" if _cuda_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(args.model_size, device=device, compute_type=compute)
    segments, _ = model.transcribe(args.wav_path, language=args.language, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    print(json.dumps({"text": text, "device": device}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
