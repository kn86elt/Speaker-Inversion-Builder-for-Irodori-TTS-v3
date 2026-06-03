#!/usr/bin/env python3
"""Run Irodori prepare_manifest.py with Windows-friendly local WAV handling.

On some Windows environments, importing hashlib before pyarrow can make
datasets/pyarrow crash with ACCESS_VIOLATION. The upstream script imports
hashlib first, so this wrapper loads datasets before delegating to it.

datasets.Audio also requires torchcodec/FFmpeg shared DLLs on recent datasets
versions. For the WebUI's local WAV JSONL, bypass that decoder and load WAVs
directly with soundfile.
"""
from __future__ import annotations

import runpy
import sys
import argparse
import math
import os
from pathlib import Path

import torch
from datasets import Audio, Value, load_dataset  # noqa: F401


def _coerce_audio_for_local_wav(audio_value):
    if isinstance(audio_value, str):
        import soundfile as sf

        data, sr = sf.read(audio_value, always_2d=True, dtype="float32")
        wav = torch.from_numpy(data).transpose(0, 1).contiguous()
        return wav, int(sr)
    if isinstance(audio_value, dict):
        if "array" not in audio_value or "sampling_rate" not in audio_value:
            raise ValueError("Audio dict must include keys: 'array', 'sampling_rate'")
        wav = torch.as_tensor(audio_value["array"]).float()
        sr = int(audio_value["sampling_rate"])
    elif hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        wav = torch.as_tensor(samples.data).float()
        sr = int(samples.sample_rate)
    elif hasattr(audio_value, "data") and hasattr(audio_value, "sample_rate"):
        wav = torch.as_tensor(audio_value.data).float()
        sr = int(audio_value.sample_rate)
    else:
        raise TypeError(f"Unsupported audio value type: {type(audio_value)}")

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        if wav.shape[1] <= 8 and wav.shape[0] > wav.shape[1]:
            wav = wav.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"Unsupported decoded audio shape: {tuple(wav.shape)}")
    if wav.numel() == 0:
        raise ValueError("Decoded audio is empty")
    return wav, sr


def _iter_rank_examples_for_local_jsonl(dataset, *, args, rank: int, world_size: int):
    start = max(0, int(args.skip_samples))
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for idx in range(start + rank, len(dataset), world_size):
            yield idx, dataset[int(idx)]
        return
    for idx, sample in enumerate(dataset):
        if idx < start or idx % world_size != rank:
            continue
        yield idx, sample


def _dataset_path_for_load_dataset(path_value: str, *, base_dir: Path) -> str:
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
        return rel.as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(path.resolve(), start=base_dir.resolve())).as_posix()
        except ValueError:
            return path.as_posix()


def _load_dataset_for_local_jsonl(path, name=None, split="train", data_files=None, **kwargs):
    if path == "json" and isinstance(data_files, dict) and split in data_files:
        data_files = data_files[split]
    if path == "json" and isinstance(data_files, str) and data_files.startswith(f"{split}="):
        data_files = data_files.split("=", 1)[1]
    if path == "json" and isinstance(data_files, (list, tuple)) and len(data_files) == 1:
        first = str(data_files[0])
        if first.startswith(f"{split}="):
            data_files = first.split("=", 1)[1]
    if path == "json" and isinstance(data_files, str):
        data_files = _dataset_path_for_load_dataset(data_files, base_dir=Path.cwd())
    if os.environ.get("IRODORI_WRAPPER_DEBUG"):
        print(f"[wrapper-debug] load_dataset data_files={data_files!r}", file=sys.stderr)
    return load_dataset(path=path, name=name, split=split, data_files=data_files, **kwargs)


def parse_optional_float_proxy(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected float or one of [none, null, off, disable, disabled], got: {value}"
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"normalize-db must be finite, got: {value}")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: prepare_manifest_wrapper.py <prepare_manifest.py> [args...]")
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    globals_dict = runpy.run_path(str(script), run_name="_irodori_prepare_manifest")
    module_globals = globals_dict["_run_worker"].__globals__
    module_globals["Audio"] = lambda *args, **kwargs: Value("string")
    module_globals["load_dataset"] = _load_dataset_for_local_jsonl
    module_globals["_coerce_audio"] = _coerce_audio_for_local_wav
    module_globals["_iter_rank_examples"] = _iter_rank_examples_for_local_jsonl
    old_argv = sys.argv
    try:
        sys.argv = [str(script)] + old_argv[2:]
        parser = globals_dict["argparse"].ArgumentParser(
            description=(
                "Precompute DACVAE latents directly from a Hugging Face dataset "
                "(without saving intermediate audio files)."
            )
        )
        _add_prepare_manifest_args(parser)
        args = parser.parse_args()
        if args.flush_every < 0:
            raise ValueError("--flush-every must be >= 0.")
        if args.num_gpus is not None and args.num_gpus < 1:
            raise ValueError("--num-gpus must be >= 1 when provided.")
        if args.skip_samples < 0:
            raise ValueError("--skip-samples must be >= 0.")
        if args.prefetch < 0:
            raise ValueError("--prefetch must be >= 0.")
        if args.prefetch_workers < 1:
            raise ValueError("--prefetch-workers must be >= 1.")
        args.speaker_columns = module_globals["_parse_speaker_columns"](args.speaker_column)
        args.speaker_id_namespace = module_globals["_resolve_speaker_namespace"](args)
        if os.environ.get("IRODORI_WRAPPER_DEBUG"):
            print(f"[wrapper-debug] raw data_files={args.data_files!r}", file=sys.stderr)
        if len(args.data_files or []) == 1 and str(args.data_files[0]).startswith(f"{args.split}="):
            source_path = str(args.data_files[0]).split("=", 1)[1]
            args.data_files = [
                f"{args.split}={_dataset_path_for_load_dataset(source_path, base_dir=script.parent)}"
            ]
        else:
            args.data_files = [
                _dataset_path_for_load_dataset(str(item), base_dir=script.parent)
                for item in (args.data_files or [])
            ]
        if os.environ.get("IRODORI_WRAPPER_DEBUG"):
            print(f"[wrapper-debug] parsed data_files={args.data_files!r}", file=sys.stderr)
        module_globals["_run_worker"](args, rank=0, world_size=1, local_rank=0)
    finally:
        sys.argv = old_argv


def _add_prepare_manifest_args(parser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", nargs="+", default=None)
    parser.add_argument("--audio-column", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--text-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--caption-column", default=None)
    parser.add_argument("--speaker-column", action="append", default=None)
    parser.add_argument("--speaker-id-prefix", default=None)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--codec-deterministic-encode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--codec-deterministic-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-db", type=globals()["parse_optional_float_proxy"], default="-16.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--shard-strategy", choices=["auto", "stride", "contiguous", "dataset"], default="auto")
    parser.add_argument("--merge-output", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--target-sample-rate", type=int, default=None)
    parser.add_argument("--min-sample-rate", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--prefetch-workers", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-all", action="store_true")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-dir", default=None)


if __name__ == "__main__":
    main()
