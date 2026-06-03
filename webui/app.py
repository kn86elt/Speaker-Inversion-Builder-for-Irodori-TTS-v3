#!/usr/bin/env python3
"""Speaker Inversion WebUI – FastAPI backend."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

# ─── Paths ────────────────────────────────────────────────────────────────────
WEBUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBUI_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SEGMENTS_DIR = DATA_DIR / "segments"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOGS_DIR = DATA_DIR / "logs"
STATE_FILE = DATA_DIR / "state.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
IMPORTS_DIR = DATA_DIR / "imports"

DATASETS_DIR = DATA_DIR / "datasets"          # dataset projects
SPEAKER_OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # trained embeddings (above data/)

for _d in (DATA_DIR, UPLOADS_DIR, SEGMENTS_DIR, OUTPUTS_DIR, LOGS_DIR, DATASETS_DIR, SPEAKER_OUTPUTS_DIR, IMPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _default_irodori_root() -> Path:
    env_root = os.environ.get("IRODORI_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        PROJECT_ROOT.parent / "Irodori-TTS-v3",
        PROJECT_ROOT / "Irodori-TTS-v3",
    ])
    for candidate in candidates:
        if (candidate / "train.py").exists():
            return candidate
    return Path(env_root) if env_root else PROJECT_ROOT.parent / "Irodori-TTS-v3"


# irodori-TTS root – overridable via settings, --irodori-root, or IRODORI_ROOT.
IRODORI_ROOT = _default_irodori_root()

# ─── State ────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()


def _empty_state() -> dict:
    return {"files": {}, "segments": [], "markers": {}}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return _empty_state()


def _save_state(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


_state: dict[str, Any] = _load_state()
_state.setdefault("files", {})
_state.setdefault("segments", [])
_state.setdefault("markers", {})


def _load_settings() -> dict[str, str]:
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open(encoding="utf-8") as f:
                data = json.load(f)
            return {
                "irodori_root": str(data.get("irodori_root", "")).strip(),
                "uv_exe": str(data.get("uv_exe", "")).strip(),
                "checkpoint_path": str(data.get("checkpoint_path", "")).strip(),
            }
        except Exception:
            pass
    return {"irodori_root": "", "uv_exe": "", "checkpoint_path": ""}


def _save_settings(settings: dict[str, str]) -> None:
    with SETTINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


_settings: dict[str, str] = _load_settings()


def _effective_irodori_root(root: str = "") -> Path:
    value = root.strip() or _settings.get("irodori_root", "").strip()
    return Path(value) if value else IRODORI_ROOT


def _effective_uv(override: str = "") -> str:
    return _find_uv(override or _settings.get("uv_exe", ""))


def _browse_directory(initial_dir: Path, title: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Folder browser is unavailable: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        selected = filedialog.askdirectory(
            parent=root,
            title=title,
            initialdir=str(initial_dir if initial_dir.exists() else PROJECT_ROOT.parent),
        )
        return str(selected or "")
    finally:
        root.destroy()


def _file_audio_path(file_id: str) -> Path | None:
    upload = UPLOADS_DIR / f"{file_id}.wav"
    if upload.exists():
        return upload
    with _state_lock:
        item = dict(_state.get("files", {}).get(file_id) or {})
    dataset_wav = item.get("dataset_wav")
    if dataset_wav:
        path = Path(dataset_wav)
        if path.exists():
            return path
    return None


def _segment_audio_path(seg: dict) -> Path | None:
    dataset_wav = seg.get("dataset_wav")
    if dataset_wav:
        path = Path(dataset_wav)
        if path.exists():
            return path
    path = SEGMENTS_DIR / f"{seg['id']}.wav"
    return path if path.exists() else None


def _safe_relpath(name: str) -> Path:
    parts = []
    for part in Path(name.replace("\\", "/")).parts:
        if part in ("", ".", "..") or part.endswith(":"):
            continue
        parts.append(part)
    return Path(*parts) if parts else Path("file")


def _copy_audio_if_exists(src: Path | None, dst: Path) -> str:
    if not src or not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(src), str(dst))
    return dst.as_posix()


async def _restore_project_state(project_state_path: Path) -> dict:
    loop = asyncio.get_event_loop()
    with project_state_path.open(encoding="utf-8") as f:
        project_state = json.load(f)

    project_name = str(project_state.get("name") or project_state_path.parent.name)
    speaker_name = str(project_state.get("speaker") or "")
    loaded_files: dict[str, dict] = {}
    loaded_segments: list[dict] = []

    def resolve_audio(item: dict) -> Path | None:
        rel_audio = item.get("dataset_wav", "")
        if rel_audio:
            path = (project_state_path.parent / rel_audio).resolve()
            if path.exists():
                return path
        return None

    for file_id, item in (project_state.get("files") or {}).items():
        item = dict(item)
        audio_path = resolve_audio(item)
        if not audio_path:
            continue
        info = await loop.run_in_executor(_thread_pool, _get_audio_info, audio_path)
        item["id"] = file_id
        item["dataset_wav"] = str(audio_path)
        item["duration"] = info.get("duration", item.get("duration", 0.0))
        item["samplerate"] = info.get("samplerate", item.get("samplerate"))
        item["channels"] = info.get("channels", item.get("channels"))
        item["subtype"] = info.get("subtype", item.get("subtype"))
        loaded_files[file_id] = item

    for seg in (project_state.get("segments") or []):
        seg = dict(seg)
        audio_path = resolve_audio(seg)
        if audio_path:
            seg["dataset_wav"] = str(audio_path)
        loaded_segments.append(seg)

    if not loaded_files:
        raise HTTPException(400, "No usable WAV files were found in project_state.json")

    with _state_lock:
        _state["files"] = loaded_files
        _state["segments"] = loaded_segments
        _state["markers"] = {
            str(fid): [float(v) for v in values if isinstance(v, (int, float))]
            for fid, values in (project_state.get("markers") or {}).items()
            if fid in loaded_files and isinstance(values, list)
        }
        _state["last_built_dataset"] = project_name
        _state["dataset_dirty"] = True
        _save_state(_state)

    return {
        "name": project_name,
        "speaker": speaker_name,
        "file_count": len(loaded_files),
        "segment_count": len(loaded_segments),
        "first_file_id": next(iter(loaded_files), ""),
    }

# ─── Thread pool ──────────────────────────────────────────────────────────────
_thread_pool = ThreadPoolExecutor(max_workers=4)

# ─── Audio utilities ──────────────────────────────────────────────────────────

def _audio_to_wav(src: Path, dst: Path) -> None:
    """Convert audio to 44.1 kHz mono WAV using ffmpeg, then pydub as fallback."""
    try:
        import subprocess as _sp
        _sp.run(
            ["ffmpeg", "-y", "-i", str(src), "-ar", "44100", "-ac", "1", str(dst)],
            capture_output=True, check=True,
        )
        return
    except Exception:
        pass
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(src)).set_channels(1).set_frame_rate(44100)
    audio.export(str(dst), format="wav")


def _get_duration(path: Path) -> float:
    try:
        import soundfile as sf
        return sf.info(str(path)).duration
    except Exception:
        return 0.0


def _get_audio_info(path: Path) -> dict:
    """Return detected format metadata via soundfile."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,       # e.g. "PCM_16"
            "format": info.format,         # e.g. "WAV"
            "duration": round(info.duration, 3),
            "frames": info.frames,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _detect_silence_markers(
    wav_path: Path,
    min_silence_ms: int,
    silence_thresh_db: float,
) -> list[float]:
    """Return midpoint positions (sec) of silence gaps between non-silent chunks."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    audio = AudioSegment.from_wav(str(wav_path))
    chunks = detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=int(silence_thresh_db))
    markers = []
    for i in range(1, len(chunks)):
        gap_start_ms = chunks[i - 1][1]
        gap_end_ms = chunks[i][0]
        mid_ms = (gap_start_ms + gap_end_ms) / 2.0
        markers.append(round(mid_ms / 1000.0, 3))
    return markers


def _do_split_silence(
    wav_path: Path,
    min_silence_ms: int,
    silence_thresh_db: float,
    keep_silence_ms: int,
) -> list[tuple[float, float]]:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    audio = AudioSegment.from_wav(str(wav_path))
    chunks = detect_nonsilent(audio, min_silence_len=min_silence_ms, silence_thresh=int(silence_thresh_db))
    total_ms = len(audio)
    out = []
    for s, e in chunks:
        s = max(0, s - keep_silence_ms)
        e = min(total_ms, e + keep_silence_ms)
        out.append((s / 1000.0, e / 1000.0))
    return out


def _extract_segment(src: Path, start_sec: float, end_sec: float, dst: Path) -> None:
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(str(src))
    seg = audio[int(start_sec * 1000): int(end_sec * 1000)]
    seg.export(str(dst), format="wav")


def _delete_audio_range(src: Path, start_sec: float, end_sec: float, dst: Path) -> float:
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(str(src))
    start_ms = max(0, int(start_sec * 1000))
    end_ms = min(len(audio), int(end_sec * 1000))
    edited = audio[:start_ms] + audio[end_ms:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    edited.export(str(dst), format="wav")
    return len(edited) / 1000.0


# ─── Whisper ──────────────────────────────────────────────────────────────────
def _transcribe_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("HF_HOME", str(DATA_DIR / "hf-cache"))
    env.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf-cache" / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(DATA_DIR / "hf-cache" / "hub"))
    ffmpeg_bin = env.get("FFMPEG_BIN", "").strip()
    if ffmpeg_bin:
        env["PATH"] = ffmpeg_bin + os.pathsep + env.get("PATH", "")
    return env


def _transcribe_error(proc: Any) -> str:
    msg = (proc.stderr or proc.stdout or "").strip()
    return msg or f"transcription failed with exit code {proc.returncode}"


def _do_transcribe_many(wav_paths: list[Path]) -> dict[str, str]:
    if not wav_paths:
        return {}
    worker = WEBUI_DIR / "transcribe_worker.py"
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    cmd = [sys.executable, str(worker), "--device", device] + [str(path) for path in wav_paths]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=_transcribe_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(_transcribe_error(proc))

    results: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = str(item.get("path") or "")
        if path:
            results[str(Path(path).resolve())] = str(item.get("text", "")).strip()
    return results


def _do_transcribe(wav_path: Path) -> str:
    results = _do_transcribe_many([wav_path])
    return results.get(str(wav_path.resolve()), "")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Speaker Inversion Builder")
app.mount("/static", StaticFiles(directory=str(WEBUI_DIR / "static")), name="static")
_templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))


@app.get("/")
async def index(request: Request):
    return _templates.TemplateResponse(request, "index.html")


# ─── Audio serving ────────────────────────────────────────────────────────────

@app.get("/audio/upload/{file_id}")
async def serve_upload(file_id: str):
    p = _file_audio_path(file_id)
    if not p:
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


@app.get("/audio/segment/{seg_id}")
async def serve_segment(seg_id: str):
    # Dataset-loaded segments store their actual WAV path in dataset_wav
    with _state_lock:
        seg = next((s for s in _state["segments"] if s["id"] == seg_id), None)
    if seg and seg.get("dataset_wav"):
        p = Path(seg["dataset_wav"])
        if p.exists():
            return FileResponse(str(p), media_type="audio/wav")
    p = SEGMENTS_DIR / f"{seg_id}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


@app.get("/audio/output/{filename}")
async def serve_output(filename: str):
    p = OUTPUTS_DIR / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


# ─── State ────────────────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    with _state_lock:
        return _state


@app.post("/api/clear_data_prep")
async def clear_data_prep():
    with _state_lock:
        _state["files"] = {}
        _state["segments"] = []
        _state["markers"] = {}
        _state["dataset_dirty"] = False
        _save_state(_state)
    for root in (UPLOADS_DIR, SEGMENTS_DIR):
        for path in root.glob("*.wav"):
            path.unlink(missing_ok=True)
    return {"ok": True}


# ─── Upload ───────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    loop = asyncio.get_event_loop()
    added = []

    for file in files:
        file_id = str(uuid.uuid4())
        suffix = Path(file.filename or "x.wav").suffix.lower()
        content = await file.read()

        tmp = UPLOADS_DIR / f"{file_id}{suffix}"
        tmp.write_bytes(content)

        wav = UPLOADS_DIR / f"{file_id}.wav"
        if suffix == ".wav":
            wav = tmp
        else:
            await loop.run_in_executor(_thread_pool, _audio_to_wav, tmp, wav)
            tmp.unlink(missing_ok=True)

        info = await loop.run_in_executor(_thread_pool, _get_audio_info, wav)

        with _state_lock:
            _state["files"][file_id] = {
                "id": file_id,
                "name": file.filename,
                "duration": info.get("duration", 0.0),
                "samplerate": info.get("samplerate"),
                "channels": info.get("channels"),
                "subtype": info.get("subtype"),
            }
            _save_state(_state)

        added.append(file_id)

    return {"added": added}


@app.post("/api/import_dataset_folder")
async def import_dataset_folder(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files selected")

    import_id = str(uuid.uuid4())
    import_root = IMPORTS_DIR / import_id
    import_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for file in files:
        rel = _safe_relpath(file.filename or file.headers.get("filename", "file"))
        dst = import_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(await file.read())
        saved_paths.append(dst)

    project_state_path = next((p for p in saved_paths if p.name == "project_state.json"), None)
    source_jsonl_path = next((p for p in saved_paths if p.name == "source.jsonl"), None)
    if not project_state_path and not source_jsonl_path:
        raise HTTPException(400, "project_state.json or source.jsonl was not found")

    loaded_files: dict[str, dict] = {}
    loaded_segments: list[dict] = []
    project_name = import_root.name
    speaker_name = ""

    if project_state_path:
        return await _restore_project_state(project_state_path)
    else:
        loop = asyncio.get_event_loop()
        project_name = source_jsonl_path.parent.name
        all_wavs = [p for p in saved_paths if p.suffix.lower() == ".wav"]
        wav_by_name = {p.name.lower(): p for p in all_wavs}
        with source_jsonl_path.open(encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        for idx, item in enumerate(entries, 1):
            audio_name = Path(str(item.get("audio", ""))).name.lower()
            audio_path = wav_by_name.get(audio_name)
            if not audio_path or not audio_path.exists():
                continue
            if not speaker_name:
                speaker_name = str(item.get("speaker", ""))
            info = await loop.run_in_executor(_thread_pool, _get_audio_info, audio_path)
            dur = float(info.get("duration") or 0.0)
            file_id = f"import_{import_id}_{idx:04d}"
            seg_id = f"import_seg_{import_id}_{idx:04d}"
            loaded_files[file_id] = {
                "id": file_id,
                "name": audio_path.name,
                "duration": round(dur, 3),
                "samplerate": info.get("samplerate"),
                "channels": info.get("channels"),
                "subtype": info.get("subtype"),
                "dataset_wav": str(audio_path),
            }
            loaded_segments.append({
                "id": seg_id,
                "file_id": file_id,
                "start": 0.0,
                "end": round(dur, 3),
                "duration": round(dur, 3),
                "text": item.get("text", ""),
                "transcribed": bool(str(item.get("text", "")).strip()),
                "dataset_wav": str(audio_path),
            })

    if not loaded_files:
        raise HTTPException(400, "No usable WAV files were found")

    with _state_lock:
        _state["files"] = loaded_files
        _state["segments"] = loaded_segments
        _state["markers"] = {}
        _state["last_built_dataset"] = project_name
        _state["dataset_dirty"] = True
        _save_state(_state)

    return {
        "name": project_name,
        "speaker": speaker_name,
        "file_count": len(loaded_files),
        "segment_count": len(loaded_segments),
        "first_file_id": next(iter(loaded_files), ""),
    }


# ─── File operations ──────────────────────────────────────────────────────────

@app.delete("/api/file/{file_id}")
async def delete_file(file_id: str):
    with _state_lock:
        del_segs = [s["id"] for s in _state["segments"] if s["file_id"] == file_id]
        _state["segments"] = [s for s in _state["segments"] if s["file_id"] != file_id]
        _state["files"].pop(file_id, None)
        _state.setdefault("markers", {}).pop(file_id, None)
        _save_state(_state)
    for sid in del_segs:
        (SEGMENTS_DIR / f"{sid}.wav").unlink(missing_ok=True)
    (UPLOADS_DIR / f"{file_id}.wav").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/file/{file_id}/as_segment")
async def file_as_segment(file_id: str):
    wav = _file_audio_path(file_id)
    if not wav:
        raise HTTPException(404)
    loop = asyncio.get_event_loop()
    dur = await loop.run_in_executor(_thread_pool, _get_duration, wav)

    seg_id = str(uuid.uuid4())
    dst = SEGMENTS_DIR / f"{seg_id}.wav"
    await loop.run_in_executor(_thread_pool, shutil.copy, str(wav), str(dst))

    seg = {
        "id": seg_id,
        "file_id": file_id,
        "start": 0.0,
        "end": round(dur, 3),
        "duration": round(dur, 3),
        "text": "",
        "transcribed": False,
    }
    with _state_lock:
        _state["segments"] = [s for s in _state["segments"] if s["file_id"] != file_id]
        _state["segments"].append(seg)
        _save_state(_state)
    return {"segment": seg}


# ─── File audio info ──────────────────────────────────────────────────────────

@app.get("/api/file/{file_id}/info")
async def get_file_audio_info(file_id: str):
    p = _file_audio_path(file_id)
    if not p:
        raise HTTPException(404)
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(_thread_pool, _get_audio_info, p)
    return info


# ─── Fix WAV header SR ────────────────────────────────────────────────────────

# ─── Auto-markers (detect silence boundaries) ─────────────────────────────────

class MarkerParams(BaseModel):
    min_silence_ms: int = 700
    silence_thresh_db: float = -40.0


class FileMarkersRequest(BaseModel):
    markers: list[float] = []


@app.post("/api/file/{file_id}/auto_markers")
async def auto_markers(file_id: str, params: MarkerParams):
    wav = _file_audio_path(file_id)
    if not wav:
        raise HTTPException(404)
    loop = asyncio.get_event_loop()
    markers = await loop.run_in_executor(
        _thread_pool,
        _detect_silence_markers, wav, params.min_silence_ms, params.silence_thresh_db,
    )
    return {"markers": markers}


@app.put("/api/file/{file_id}/markers")
async def save_file_markers(file_id: str, req: FileMarkersRequest):
    with _state_lock:
        if file_id not in _state.get("files", {}):
            raise HTTPException(404)
        markers = sorted({round(float(t), 3) for t in req.markers if float(t) > 0})
        if markers:
            _state.setdefault("markers", {})[file_id] = markers
        else:
            _state.setdefault("markers", {}).pop(file_id, None)
        _save_state(_state)
    return {"markers": markers}


# ─── Extract range as segment ─────────────────────────────────────────────────

class ExtractRangeRequest(BaseModel):
    start: float
    end: float


@app.post("/api/file/{file_id}/extract_range")
async def extract_range(file_id: str, req: ExtractRangeRequest):
    wav = _file_audio_path(file_id)
    if not wav:
        raise HTTPException(404)
    if req.end <= req.start:
        raise HTTPException(400, "end must be greater than start")
    loop = asyncio.get_event_loop()
    seg_id = str(uuid.uuid4())
    dst = SEGMENTS_DIR / f"{seg_id}.wav"
    await loop.run_in_executor(_thread_pool, _extract_segment, wav, req.start, req.end, dst)
    seg = {
        "id": seg_id,
        "file_id": file_id,
        "start": round(req.start, 3),
        "end": round(req.end, 3),
        "duration": round(req.end - req.start, 3),
        "text": "",
        "transcribed": False,
    }
    with _state_lock:
        _state["segments"].append(seg)
        _save_state(_state)
    return {"segment": seg}


# ─── Split on silence ─────────────────────────────────────────────────────────

@app.post("/api/file/{file_id}/delete_range")
async def delete_range(file_id: str, req: ExtractRangeRequest):
    wav = _file_audio_path(file_id)
    if not wav:
        raise HTTPException(404)
    if req.end <= req.start:
        raise HTTPException(400, "end must be greater than start")

    loop = asyncio.get_event_loop()
    dst = UPLOADS_DIR / f"{file_id}.wav"
    new_duration = await loop.run_in_executor(_thread_pool, _delete_audio_range, wav, req.start, req.end, dst)
    info = await loop.run_in_executor(_thread_pool, _get_audio_info, dst)
    delta = req.end - req.start

    removed_ids: list[str] = []
    adjusted_segments: list[dict] = []
    with _state_lock:
        item = _state["files"].get(file_id)
        if item:
            item["duration"] = info.get("duration", round(new_duration, 3))
            item["samplerate"] = info.get("samplerate", item.get("samplerate"))
            item["channels"] = info.get("channels", item.get("channels"))
            item["subtype"] = info.get("subtype", item.get("subtype"))
            item["dataset_wav"] = str(dst)

        for seg in _state["segments"]:
            if seg.get("file_id") != file_id:
                adjusted_segments.append(seg)
                continue
            if seg.get("end", 0) <= req.start:
                adjusted_segments.append(seg)
                continue
            if seg.get("start", 0) >= req.end:
                seg = dict(seg)
                seg["start"] = round(seg["start"] - delta, 3)
                seg["end"] = round(seg["end"] - delta, 3)
                adjusted_segments.append(seg)
                continue
            removed_ids.append(seg["id"])

        _state["segments"] = adjusted_segments
        _state["dataset_dirty"] = True
        _save_state(_state)

    for sid in removed_ids:
        (SEGMENTS_DIR / f"{sid}.wav").unlink(missing_ok=True)

    return {"duration": round(new_duration, 3), "removed_segments": removed_ids}


class SilenceParams(BaseModel):
    min_silence_ms: int = 700
    silence_thresh_db: float = -40.0
    keep_silence_ms: int = 200


@app.post("/api/file/{file_id}/split_silence")
async def split_file_silence(file_id: str, params: SilenceParams):
    wav = _file_audio_path(file_id)
    if not wav:
        raise HTTPException(404)
    loop = asyncio.get_event_loop()

    regions = await loop.run_in_executor(
        _thread_pool,
        _do_split_silence, wav,
        params.min_silence_ms, params.silence_thresh_db, params.keep_silence_ms,
    )
    if not regions:
        return {"segments": [], "message": "No non-silent regions found"}

    seg_data: list[tuple[str, float, float]] = []
    for start, end in regions:
        seg_id = str(uuid.uuid4())
        dst = SEGMENTS_DIR / f"{seg_id}.wav"
        await loop.run_in_executor(_thread_pool, _extract_segment, wav, start, end, dst)
        seg_data.append((seg_id, start, end))

    new_segs = []
    with _state_lock:
        _state["segments"] = [s for s in _state["segments"] if s["file_id"] != file_id]
        for seg_id, start, end in seg_data:
            seg = {
                "id": seg_id,
                "file_id": file_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "text": "",
                "transcribed": False,
            }
            _state["segments"].append(seg)
            new_segs.append(seg)
        _save_state(_state)
    return {"segments": new_segs}


# ─── Manual split ─────────────────────────────────────────────────────────────

class ManualSplitParams(BaseModel):
    positions: list[float]  # split times in seconds within the segment


@app.post("/api/segment/{seg_id}/split")
async def manual_split_segment(seg_id: str, params: ManualSplitParams):
    with _state_lock:
        seg = next((s for s in _state["segments"] if s["id"] == seg_id), None)
        if seg is None:
            raise HTTPException(404)
        seg = dict(seg)

    dur = seg["end"] - seg["start"]
    positions = sorted(p for p in params.positions if 0 < p < dur)
    if not positions:
        raise HTTPException(400, "No valid split positions")

    boundaries = [0.0] + positions + [dur]
    loop = asyncio.get_event_loop()
    src = _segment_audio_path(seg)
    if not src:
        raise HTTPException(404)

    seg_data: list[tuple[str, float, float]] = []
    source_offset = 0.0
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        new_id = str(uuid.uuid4())
        dst = SEGMENTS_DIR / f"{new_id}.wav"
        await loop.run_in_executor(_thread_pool, _extract_segment, src, source_offset + s, source_offset + e, dst)
        abs_start = seg["start"] + s
        abs_end = seg["start"] + e
        seg_data.append((new_id, abs_start, abs_end))

    new_segs = []
    with _state_lock:
        idx = next((i for i, s in enumerate(_state["segments"]) if s["id"] == seg_id), None)
        if idx is None:
            raise HTTPException(404)
        new_seg_objs = []
        for j, (new_id, abs_start, abs_end) in enumerate(seg_data):
            s = {
                "id": new_id,
                "file_id": seg["file_id"],
                "start": round(abs_start, 3),
                "end": round(abs_end, 3),
                "duration": round(abs_end - abs_start, 3),
                "text": seg["text"] if j == 0 else "",
                "transcribed": seg["transcribed"] if j == 0 else False,
            }
            new_seg_objs.append(s)
            new_segs.append(s)
        _state["segments"][idx : idx + 1] = new_seg_objs
        _save_state(_state)

    src.unlink(missing_ok=True)
    return {"segments": new_segs}


# ─── Transcription ────────────────────────────────────────────────────────────

@app.post("/api/segment/{seg_id}/transcribe")
async def transcribe_segment(seg_id: str):
    with _state_lock:
        seg = next((s for s in _state["segments"] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404)
    p = _segment_audio_path(seg)
    if not p:
        raise HTTPException(404)
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(_thread_pool, _do_transcribe, p)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    with _state_lock:
        seg = next((s for s in _state["segments"] if s["id"] == seg_id), None)
        if seg:
            seg["text"] = text
            seg["transcribed"] = True
        _save_state(_state)
    return {"text": text}


@app.post("/api/transcribe_all")
async def transcribe_all_segments():
    with _state_lock:
        pending = [dict(s) for s in _state["segments"] if not s.get("transcribed")]

    async def stream():
        loop = asyncio.get_event_loop()
        total = len(pending)
        yield f"data: {json.dumps({'total': total})}\n\n"

        jobs: list[tuple[dict, Path]] = []
        missing: set[str] = set()
        for seg in pending:
            p = _segment_audio_path(seg)
            if p:
                jobs.append((seg, p))
            else:
                missing.add(seg["id"])

        seg_by_path: dict[str, dict] = {str(p.resolve()): seg for seg, p in jobs}
        wav_paths = [p for _, p in jobs]
        queue: asyncio.Queue = asyncio.Queue()

        def run_worker():
            if not wav_paths:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                return
            worker = WEBUI_DIR / "transcribe_worker.py"
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            cmd = [sys.executable, str(worker), "--device", device] + [str(p) for p in wav_paths]
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=_transcribe_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        loop.call_soon_threadsafe(queue.put_nowait, ("line", line))
                proc.wait()
                if proc.returncode != 0:
                    err = (proc.stderr.read() or "").strip() or f"exit code {proc.returncode}"
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", err))
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        _thread_pool.submit(run_worker)

        progress = 0
        failed = 0
        completed_ids: set[str] = set()

        for seg in pending:
            if seg["id"] in missing:
                failed += 1
                progress += 1
                completed_ids.add(seg["id"])
                yield f"data: {json.dumps({'id': seg['id'], 'error': 'file missing', 'progress': progress, 'total': total})}\n\n"

        while True:
            kind, data = await queue.get()
            if kind == "error":
                for seg in pending:
                    if seg["id"] not in completed_ids:
                        failed += 1
                        progress += 1
                        yield f"data: {json.dumps({'id': seg['id'], 'error': data, 'progress': progress, 'total': total})}\n\n"
                break
            elif kind == "done":
                break
            elif kind == "line":
                try:
                    item = json.loads(data)
                    path_str = str(Path(str(item.get("path") or "")).resolve())
                    seg = seg_by_path.get(path_str)
                    if seg and seg["id"] not in completed_ids:
                        text = str(item.get("text", "")).strip()
                        progress += 1
                        completed_ids.add(seg["id"])
                        with _state_lock:
                            s = next((x for x in _state["segments"] if x["id"] == seg["id"]), None)
                            if s:
                                s["text"] = text
                                s["transcribed"] = True
                            _save_state(_state)
                        yield f"data: {json.dumps({'id': seg['id'], 'text': text, 'progress': progress, 'total': total})}\n\n"
                except Exception:
                    pass

        yield f"data: {json.dumps({'done': True, 'failed': failed})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─── Segment CRUD ─────────────────────────────────────────────────────────────

class UpdateTextRequest(BaseModel):
    text: str


@app.put("/api/segment/{seg_id}/text")
async def update_segment_text(seg_id: str, req: UpdateTextRequest):
    with _state_lock:
        seg = next((s for s in _state["segments"] if s["id"] == seg_id), None)
        if seg is None:
            raise HTTPException(404)
        seg["text"] = req.text
        seg["transcribed"] = bool(req.text.strip())
        _state["dataset_dirty"] = True
        _save_state(_state)
    return {"ok": True}


@app.delete("/api/segment/{seg_id}")
async def delete_segment(seg_id: str):
    with _state_lock:
        _state["segments"] = [s for s in _state["segments"] if s["id"] != seg_id]
        _state["dataset_dirty"] = True
        _save_state(_state)
    (SEGMENTS_DIR / f"{seg_id}.wav").unlink(missing_ok=True)
    return {"ok": True}


# ─── Dataset build ────────────────────────────────────────────────────────────

class BuildDatasetRequest(BaseModel):
    job_name: str
    speaker_name: str = ""
    overwrite: bool = False
    # Training params for script generation
    device: str = "auto"
    precision: str = "bf16"
    max_steps: int = 3000
    batch_size: int = 16
    grad_accum: int = 1
    tokens: int = 16
    learning_rate: float = 0.01
    save_every: int = 250
    normalize_db: str = "-16.0"
    init_embedding: str = ""
    irodori_root: str = ""
    uv_exe: str = ""


@app.post("/api/build_dataset")
async def build_dataset(req: BuildDatasetRequest):
    with _state_lock:
        segments = [s for s in _state["segments"] if s.get("text", "").strip()]

    if not segments:
        raise HTTPException(400, "No segments with text found")

    speaker = req.speaker_name.strip()
    name = req.job_name.strip() or speaker or "dataset"
    speaker = speaker or name
    irodori = _effective_irodori_root(req.irodori_root)
    uv = _effective_uv(req.uv_exe)

    dataset_dir = DATASETS_DIR / name
    if dataset_dir.exists() and not req.overwrite:
        raise HTTPException(409, f"Dataset '{name}' already exists. Send overwrite=true to replace.")

    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    count = 0
    for i, seg in enumerate(segments):
        src = _segment_audio_path(seg)
        if not src:
            continue
        dst = wavs_dir / f"{i + 1:04d}.wav"
        shutil.copy(str(src), str(dst))
        audio_path = dst
        entries.append({
            "audio": str(audio_path.resolve()),
            "text": seg["text"],
            "speaker": speaker,
        })
        count += 1

    if not count:
        raise HTTPException(400, "No valid WAV files found")

    source_jsonl = dataset_dir / "source.jsonl"
    with source_jsonl.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with _state_lock:
        state_files = {fid: dict(f) for fid, f in _state.get("files", {}).items()}
        state_segments = [dict(s) for s in _state.get("segments", [])]
        state_markers = {
            fid: list(values)
            for fid, values in (_state.get("markers") or {}).items()
            if fid in state_files and isinstance(values, list)
        }

    saved_files: dict[str, dict] = {}
    for idx, (file_id, item) in enumerate(state_files.items(), 1):
        src = _file_audio_path(file_id)
        rel = Path("originals") / f"{idx:04d}.wav"
        copied = _copy_audio_if_exists(src, dataset_dir / rel)
        saved = dict(item)
        if copied:
            saved["dataset_wav"] = rel.as_posix()
        saved_files[file_id] = saved

    saved_segments: list[dict] = []
    for idx, seg in enumerate(state_segments, 1):
        saved = dict(seg)
        src = _segment_audio_path(seg)
        if src:
            rel = Path("project_segments") / f"{idx:04d}.wav"
            copied = _copy_audio_if_exists(src, dataset_dir / rel)
            if copied:
                saved["dataset_wav"] = rel.as_posix()
        saved_segments.append(saved)

    project_state = {
        "version": 1,
        "name": name,
        "speaker": speaker,
        "files": saved_files,
        "segments": saved_segments,
        "markers": state_markers,
    }
    with (dataset_dir / "project_state.json").open("w", encoding="utf-8") as f:
        json.dump(project_state, f, ensure_ascii=False, indent=2)

    train_params = {
        "device": req.device, "precision": req.precision,
        "max_steps": req.max_steps, "batch_size": req.batch_size,
        "grad_accum": req.grad_accum, "tokens": req.tokens,
        "learning_rate": req.learning_rate, "save_every": req.save_every,
        "normalize_db": req.normalize_db, "init_embedding": req.init_embedding,
    }
    _write_train_scripts(dataset_dir, name, irodori, uv, train_params)

    with _state_lock:
        _state["last_built_dataset"] = name
        _state["dataset_dirty"] = False
        _save_state(_state)

    return {
        "name": name,
        "count": count,
        "dataset_dir": str(dataset_dir),
        "wavs_dir": str(wavs_dir),
        "source_jsonl": str(source_jsonl),
        "train_bat": str(dataset_dir / "train.bat"),
        "train_sh": str(dataset_dir / "train.sh"),
    }


# ─── Dataset list & load ──────────────────────────────────────────────────────

@app.get("/api/datasets")
async def list_datasets():
    if not DATASETS_DIR.exists():
        return {"datasets": []}
    datasets = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        source = d / "source.jsonl"
        manifest = d / "manifest.jsonl"
        embed = SPEAKER_OUTPUTS_DIR / d.name / "checkpoint_final.speaker.safetensors"
        count = 0
        if source.exists():
            with source.open(encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        datasets.append({
            "name": d.name,
            "has_source": source.exists(),
            "has_manifest": manifest.exists(),
            "has_embedding": embed.exists(),
            "segment_count": count,
            "embedding_path": str(embed) if embed.exists() else None,
        })
    return {"datasets": datasets}


@app.post("/api/datasets/{name}/load")
async def load_dataset_project(name: str):
    dataset_dir = DATASETS_DIR / name
    project_state_path = dataset_dir / "project_state.json"
    if project_state_path.exists():
        return await _restore_project_state(project_state_path)

    source_jsonl = dataset_dir / "source.jsonl"
    if not source_jsonl.exists():
        raise HTTPException(404, f"Dataset '{name}' not found")

    loop = asyncio.get_event_loop()
    project_prefix = f"__dataset__{name}"
    files: dict[str, dict] = {}
    segments: list[dict] = []

    with source_jsonl.open(encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        item = json.loads(line)
        wav_path = Path(item["audio"])
        if not wav_path.exists():
            continue
        info = await loop.run_in_executor(_thread_pool, _get_audio_info, wav_path)
        dur = float(info.get("duration") or 0.0)
        file_id = f"{project_prefix}_{i:04d}"
        seg_id = f"__ds_{name}_{i:04d}"
        files[file_id] = {
            "id": file_id,
            "name": f"[Dataset] {name} / {wav_path.name}",
            "duration": round(dur, 3),
            "samplerate": info.get("samplerate"),
            "channels": info.get("channels"),
            "subtype": info.get("subtype"),
            "dataset_project": name,
            "dataset_wav": str(wav_path),
        }
        segments.append({
            "id": seg_id,
            "file_id": file_id,
            "start": 0.0,
            "end": round(dur, 3),
            "duration": round(dur, 3),
            "text": item.get("text", ""),
            "transcribed": bool(item.get("text", "").strip()),
            "dataset_project": name,
            "dataset_wav": str(wav_path),
        })

    with _state_lock:
        _state["files"] = {
            fid: f for fid, f in _state["files"].items()
            if f.get("dataset_project") != name
        }
        _state["files"].update(files)
        _state["segments"] = [
            s for s in _state["segments"]
            if s.get("dataset_project") != name
        ] + segments
        _state["markers"] = {
            fid: values for fid, values in (_state.get("markers") or {}).items()
            if fid in _state["files"]
        }
        _state["last_built_dataset"] = name
        _state["dataset_dirty"] = False
        _save_state(_state)

    first_file_id = next(iter(files), "")
    return {"name": name, "segment_count": len(segments), "file_count": len(files), "first_file_id": first_file_id}


# ─── Runs list (speaker embeddings) ──────────────────────────────────────────

@app.get("/api/runs")
async def list_runs():
    if not SPEAKER_OUTPUTS_DIR.exists():
        return {"runs": []}
    runs = []
    for d in sorted(SPEAKER_OUTPUTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        embed = d / "checkpoint_final.speaker.safetensors"
        step_files = sorted(
            [f for f in d.glob("checkpoint_*.speaker.safetensors")
             if f.stem != "checkpoint_final"],
            key=lambda f: int(m.group(1)) if (m := re.search(r"_(\d+)$", f.stem)) else 0,
            reverse=True,
        )
        runs.append({
            "name": d.name,
            "has_embedding": embed.exists(),
            "embedding_path": str(embed) if embed.exists() else None,
            "step_embeddings": [{"name": f.name, "path": str(f)} for f in step_files],
        })
    return {"runs": runs}


# ─── Settings ────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    saved_root = _settings.get("irodori_root", "").strip()
    root_for_defaults = Path(saved_root) if saved_root else IRODORI_ROOT
    return {
        "irodori_root": saved_root,
        "suggested_irodori_root": str(IRODORI_ROOT),
        "default_checkpoint": str(root_for_defaults / "Irodori-TTS-500M-v3" / "model.safetensors") if saved_root else "",
        "default_config": str(root_for_defaults / "configs" / "train_500m_v3_speaker_inversion.yaml") if saved_root else "",
        "uv_exe": _settings.get("uv_exe", "").strip() or _find_uv(),
        "checkpoint_path": _settings.get("checkpoint_path", "").strip(),
    }


class SettingsRequest(BaseModel):
    irodori_root: str = ""
    uv_exe: str = ""
    checkpoint_path: str = ""


@app.post("/api/settings")
async def save_settings(req: SettingsRequest):
    root = req.irodori_root.strip()
    if not root:
        raise HTTPException(400, "Irodori-TTS root path is required")
    irodori = Path(root)
    if not irodori.exists() or not irodori.is_dir():
        raise HTTPException(400, f"Irodori-TTS root does not exist: {root}")
    if not (irodori / "train.py").exists():
        raise HTTPException(400, f"train.py was not found under: {root}")

    checkpoint = req.checkpoint_path.strip()
    if not checkpoint:
        checkpoint = str(irodori / "Irodori-TTS-500M-v3" / "model.safetensors")

    _settings.update({
        "irodori_root": root,
        "uv_exe": req.uv_exe.strip(),
        "checkpoint_path": checkpoint,
    })
    _save_settings(_settings)
    return {"ok": True, **_settings}


@app.get("/api/browse/irodori_root")
async def browse_irodori_root():
    saved_root = _settings.get("irodori_root", "").strip()
    initial = Path(saved_root) if saved_root else IRODORI_ROOT
    loop = asyncio.get_event_loop()
    try:
        selected = await loop.run_in_executor(
            _thread_pool,
            _browse_directory,
            initial,
            "Select Irodori-TTS v3 folder",
        )
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"path": selected}


# ─── uv helper ───────────────────────────────────────────────────────────────

def _find_uv(override: str = "") -> str:
    """Return path to uv binary; use override when provided."""
    if override.strip():
        return override.strip()
    import shutil
    found = shutil.which("uv")
    if found:
        return found
    # Common Windows install path
    candidate = Path.home() / ".local" / "bin" / "uv.exe"
    if candidate.exists():
        return str(candidate)
    return "uv"  # last-resort: hope it's in PATH


def _uv_python(uv: str) -> list[str]:
    """Build a uv Python command that preserves the existing Irodori venv."""
    return [uv, "run", "--no-sync", "python"]


def _uv_run(uv: str, irodori: Path, script: str) -> list[str]:
    """Build  uv run --no-sync python <irodori/script>  prefix.

    --no-sync: preserve the venv as-is after the user selected an Irodori extra.
    The caller appends script-specific arguments.
    """
    return _uv_python(uv) + [str(irodori / script)]


_DEVICE_DETECT_CODE = (
    "import torch\n"
    "device='cpu'\n"
    "if torch.cuda.is_available():\n"
    "    device='cuda'\n"
    "elif hasattr(torch, 'xpu') and torch.xpu.is_available():\n"
    "    device='xpu'\n"
    "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
    "    device='mps'\n"
    "print(device)\n"
)


def _normalise_device(device: str | None) -> str:
    raw = (device or "auto").strip().lower()
    if raw in {"", "auto", "best"}:
        return "auto"
    if raw in {"cuda", "xpu", "mps", "cpu"}:
        return raw
    return raw


def _subprocess_env(irodori: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["UV_CACHE_DIR"] = str(DATA_DIR / "uv-cache")
    env["HF_HOME"] = str(DATA_DIR / "hf-cache")
    env["HF_DATASETS_CACHE"] = str(DATA_DIR / "hf-cache" / "datasets")
    env["HF_HUB_CACHE"] = str(DATA_DIR / "hf-cache" / "hub")
    ffmpeg_bin = env.get("FFMPEG_BIN", "").strip()
    if ffmpeg_bin:
        env["PATH"] = ffmpeg_bin + os.pathsep + env.get("PATH", "")
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    return env


def _log_path(kind: str, name: str) -> Path:
    from datetime import datetime

    safe_kind = "".join(c if c.isalnum() or c in "._-" else "_" for c in kind) or "run"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name) or "job"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{safe_kind}_{safe_name}_{stamp}.log"


def _append_log_line(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


async def _detect_device(uv: str, irodori: Path) -> str:
    proc = await asyncio.create_subprocess_exec(
        *(_uv_python(uv) + ["-c", _DEVICE_DETECT_CODE]),
        cwd=str(irodori),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_subprocess_env(irodori),
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        detected = stdout.decode("utf-8", errors="replace").strip().splitlines()
        if detected:
            return _normalise_device(detected[-1])
    return "cpu"


def _write_train_scripts(
    dataset_dir: Path,
    name: str,
    irodori: Path,
    uv: str,
    train_params: dict,
) -> None:
    """Generate train.bat and train.sh for standalone re-execution."""
    from datetime import datetime

    source_jsonl = dataset_dir / "source.jsonl"
    manifest = dataset_dir / "manifest.jsonl"
    latent_dir = dataset_dir / "latents"
    checkpoint = irodori / "Irodori-TTS-500M-v3" / "model.safetensors"
    config = irodori / "configs" / "train_500m_v3_speaker_inversion.yaml"
    output_dir = SPEAKER_OUTPUTS_DIR / name
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    p = train_params
    device   = _normalise_device(p.get("device", "auto"))
    prec     = p.get("precision", "bf16")
    steps    = p.get("max_steps", 3000)
    bs       = p.get("batch_size", 16)
    accum    = p.get("grad_accum", 1)
    tokens   = p.get("tokens", 16)
    lr       = p.get("learning_rate", 0.01)
    save_ev  = p.get("save_every", 250)
    norm_db  = p.get("normalize_db", "-16.0")
    init_emb = p.get("init_embedding", "")

    # ── train.bat (Windows) ──────────────────────────────────────────────────
    bat = [
        "@echo off",
        "chcp 65001 >nul",
        "setlocal",
        ":: Speaker Inversion - standalone training script",
        f":: Dataset   : {name}",
        f":: Generated : {now}",
        ":: Edit the variables below to adjust parameters.",
        "",
        f'set "UV={uv}"',
        'set "SCRIPT_DIR=%~dp0"',
        'for %%I in ("%SCRIPT_DIR%..\\..\\..") do set "WEBUI_ROOT=%%~fI"',
        f'set "IRODORI={irodori}"',
        'set "SOURCE_JSONL=%SCRIPT_DIR%source.jsonl"',
        'set "MANIFEST=%SCRIPT_DIR%manifest.jsonl"',
        'set "LATENT_DIR=%SCRIPT_DIR%latents"',
        f'set "CHECKPOINT={checkpoint}"',
        f'set "CONFIG={config}"',
        f'set "OUTPUT_DIR=%WEBUI_ROOT%\\outputs\\{name}"',
        f'set "SPEAKER_ID={name}"',
        "rem Optional: set this to a full shared FFmpeg build bin directory; torchcodec may require the DLLs.",
        'set "FFMPEG_BIN="',
        'if not "%FFMPEG_BIN%"=="" set "PATH=%FFMPEG_BIN%;%PATH%"',
        'set "UV_CACHE_DIR=%IRODORI%\\.uv-cache"',
        'set "HF_HOME=%IRODORI%\\.hf-cache"',
        'set "HF_DATASETS_CACHE=%IRODORI%\\.hf-cache\\datasets"',
        'set "HF_HUB_CACHE=%IRODORI%\\.hf-cache\\hub"',
        'set "PYTHONUTF8=1"',
        'set "PYTHONUNBUFFERED=1"',
        'set "PYTHONFAULTHANDLER=1"',
        "",
        ":: ---- Training parameters ----",
        f'set "DEVICE={device}"',
        f'set "PRECISION={prec}"',
        f'set "MAX_STEPS={steps}"',
        f'set "BATCH_SIZE={bs}"',
        f'set "GRAD_ACCUM={accum}"',
        f'set "TOKENS={tokens}"',
        f'set "LR={lr}"',
        f'set "SAVE_EVERY={save_ev}"',
        f'set "NORMALIZE_DB={norm_db}"',
        "",
        'if not exist "%UV%" ( echo [ERROR] uv not found: "%UV%" & pause & exit /b 1 )',
        'if not exist "%IRODORI%\\train.py" ( echo [ERROR] Irodori root is invalid: "%IRODORI%" & pause & exit /b 1 )',
        'if not exist "%WEBUI_ROOT%\\webui\\prepare_manifest_wrapper.py" ( echo [ERROR] prepare_manifest_wrapper.py not found under "%WEBUI_ROOT%\\webui" & pause & exit /b 1 )',
        'if not exist "%SOURCE_JSONL%" ( echo [ERROR] source manifest not found: "%SOURCE_JSONL%" & pause & exit /b 1 )',
        'if not exist "%CHECKPOINT%" ( echo [ERROR] checkpoint not found: "%CHECKPOINT%" & pause & exit /b 1 )',
        'cd /d "%IRODORI%" || ( echo [ERROR] Failed to enter Irodori root: "%IRODORI%" & pause & exit /b 1 )',
        "",
        'set "DETECT_PY=%TEMP%\\irodori_detect_device_%RANDOM%.py"',
        '> "%DETECT_PY%" echo import torch',
        '>> "%DETECT_PY%" echo device="cpu"',
        '>> "%DETECT_PY%" echo if torch.cuda.is_available(): device="cuda"',
        '>> "%DETECT_PY%" echo elif hasattr(torch, "xpu") and torch.xpu.is_available(): device="xpu"',
        '>> "%DETECT_PY%" echo elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device="mps"',
        '>> "%DETECT_PY%" echo print(device)',
        'if /i "%DEVICE%"=="auto" (',
        '  for /f "delims=" %%D in (\'"%UV%" run --no-sync python "%DETECT_PY%"\') do set "DEVICE=%%D"',
        ')',
        'del "%DETECT_PY%" >nul 2>&1',
        'if /i "%DEVICE%"=="auto" set "DEVICE=cpu"',
        'echo [INFO] Device: %DEVICE%',
        'echo [INFO] If audio decoding fails on Windows, install/pass a full shared FFmpeg build and set FFMPEG_BIN.',
        "",
        "echo [1/2] Preparing manifest...",
        'set "PREP_DEVICE=%DEVICE%"',
        ':prepare_manifest',
        '"%UV%" run --no-sync python "%WEBUI_ROOT%\\webui\\prepare_manifest_wrapper.py" "%IRODORI%\\prepare_manifest.py" ^',
        '  --dataset json ^',
        '  --data-files "train=%SOURCE_JSONL%" ^',
        '  --split train ^',
        '  --audio-column audio ^',
        '  --text-column text ^',
        '  --speaker-column speaker ^',
        '  --speaker-id-prefix "%SPEAKER_ID%" ^',
        '  --output-manifest "%MANIFEST%" ^',
        '  --latent-dir "%LATENT_DIR%" ^',
        '  --device %PREP_DEVICE% ^',
        '  --normalize-db %NORMALIZE_DB%',
        'set "PREP_EXIT=%ERRORLEVEL%"',
        'if not "%PREP_EXIT%"=="0" (',
        '  if /i not "%PREP_DEVICE%"=="cpu" (',
        '    echo [WARN] Manifest preparation failed on %PREP_DEVICE%; retrying on cpu.',
        '    set "PREP_DEVICE=cpu"',
        '    goto prepare_manifest',
        '  )',
        '  echo [ERROR] Manifest failed. exit_code=%PREP_EXIT%',
        '  echo [HINT] On Windows, datasets/torchcodec requires FFmpeg full shared DLLs on PATH.',
        '  pause',
        '  exit /b %PREP_EXIT%',
        ')',
        'if not exist "%MANIFEST%" ( echo [ERROR] Manifest was not created & pause & exit /b 1 )',
        'for %%I in ("%MANIFEST%") do set "MANIFEST_SIZE=%%~zI"',
        'if "%MANIFEST_SIZE%"=="0" ( echo [ERROR] Manifest is empty & pause & exit /b 1 )',
        "",
        "echo [2/2] Training...",
        '"%UV%" run --no-sync python "%IRODORI%\\train.py" ^',
        '  --config "%CONFIG%" ^',
        '  --manifest "%MANIFEST%" ^',
        '  --init-checkpoint "%CHECKPOINT%" ^',
        '  --output-dir "%OUTPUT_DIR%" ^',
        '  --device %DEVICE% ^',
        '  --precision %PRECISION% ^',
        '  --max-steps %MAX_STEPS% ^',
        '  --batch-size %BATCH_SIZE% ^',
        '  --gradient-accumulation-steps %GRAD_ACCUM% ^',
        '  --speaker-inversion-tokens %TOKENS% ^',
        '  --lr %LR% ^',
        '  --save-every %SAVE_EVERY% ^',
        '  --text-condition-dropout 0.0 ^',
        '  --speaker-condition-dropout 0.0 ^',
        '  --caption-condition-dropout 0.0 ^',
        '  --duration-speaker-dropout 0.0',
    ]
    if init_emb:
        bat[-1] += " ^"
        bat.append(f'  --speaker-inversion-init-embedding "{init_emb}"')
    bat += [
        'set "TRAIN_EXIT=%ERRORLEVEL%"',
        'if not "%TRAIN_EXIT%"=="0" ( echo [ERROR] Training failed. exit_code=%TRAIN_EXIT% & pause & exit /b %TRAIN_EXIT% )',
        "",
        f'echo Done. Embedding: %OUTPUT_DIR%\\checkpoint_final.speaker.safetensors',
        "pause",
    ]
    (dataset_dir / "train.bat").write_text("\r\n".join(bat), encoding="cp932", newline="")

    # ── train.sh (Unix / WSL) ────────────────────────────────────────────────
    sh = [
        "#!/usr/bin/env bash",
        "# Speaker Inversion – standalone training script",
        f"# Dataset   : {name}",
        f"# Generated : {now}",
        "set -e",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'WEBUI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"',
        f'UV="{uv}"',
        f'IRODORI="{irodori}"',
        'SOURCE_JSONL="$SCRIPT_DIR/source.jsonl"',
        'MANIFEST="$SCRIPT_DIR/manifest.jsonl"',
        'LATENT_DIR="$SCRIPT_DIR/latents"',
        f'CHECKPOINT="{checkpoint}"',
        f'CONFIG="{config}"',
        f'OUTPUT_DIR="$WEBUI_ROOT/outputs/{name}"',
        f'SPEAKER_ID="{name}"',
        'FFMPEG_BIN="${FFMPEG_BIN:-}"',
        'if [ -n "$FFMPEG_BIN" ]; then export PATH="$FFMPEG_BIN:$PATH"; fi',
        'export UV_CACHE_DIR="${UV_CACHE_DIR:-$IRODORI/.uv-cache}"',
        'export HF_HOME="${HF_HOME:-$IRODORI/.hf-cache}"',
        'export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$IRODORI/.hf-cache/datasets}"',
        'export HF_HUB_CACHE="${HF_HUB_CACHE:-$IRODORI/.hf-cache/hub}"',
        'export PYTHONUTF8=1',
        'export PYTHONUNBUFFERED=1',
        'export PYTHONFAULTHANDLER=1',
        "",
        "# ---- Training parameters ----",
        f"DEVICE={device}",
        f"PRECISION={prec}",
        f"MAX_STEPS={steps}",
        f"BATCH_SIZE={bs}",
        f"GRAD_ACCUM={accum}",
        f"TOKENS={tokens}",
        f"LR={lr}",
        f"SAVE_EVERY={save_ev}",
        f"NORMALIZE_DB={norm_db}",
        "",
        'if [ "$DEVICE" = "auto" ]; then',
        '  DEVICE="$("$UV" run --no-sync python - <<\'PY\'',
        'import torch',
        'device = "cpu"',
        'if torch.cuda.is_available():',
        '    device = "cuda"',
        'elif hasattr(torch, "xpu") and torch.xpu.is_available():',
        '    device = "xpu"',
        'elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():',
        '    device = "mps"',
        'print(device)',
        'PY',
        ')"',
        'fi',
        '[ -n "$DEVICE" ] || DEVICE=cpu',
        '[ "$DEVICE" != "auto" ] || DEVICE=cpu',
        'echo "[INFO] Device: $DEVICE"',
        'echo "[INFO] If audio decoding fails, install/pass a full shared FFmpeg build and set FFMPEG_BIN."',
        "",
        'echo "[1/2] Preparing manifest..."',
        'PREP_DEVICE="$DEVICE"',
        'while true; do',
        'set +e',
        '"$UV" run --no-sync python "$IRODORI/prepare_manifest.py" \\',
        '  --dataset json \\',
        '  --data-files "train=$SOURCE_JSONL" \\',
        '  --split train \\',
        '  --audio-column audio \\',
        '  --text-column text \\',
        '  --speaker-column speaker \\',
        '  --speaker-id-prefix "$SPEAKER_ID" \\',
        '  --output-manifest "$MANIFEST" \\',
        '  --latent-dir "$LATENT_DIR" \\',
        '  --device "$PREP_DEVICE" \\',
        '  --normalize-db "$NORMALIZE_DB"',
        '  PREP_EXIT=$?',
        '  set -e',
        '  if [ "$PREP_EXIT" -eq 0 ]; then break; fi',
        '  if [ "$PREP_DEVICE" != "cpu" ]; then',
        '    echo "[WARN] Manifest preparation failed on $PREP_DEVICE; retrying on cpu."',
        '    PREP_DEVICE=cpu',
        '    continue',
        '  fi',
        '  echo "[ERROR] Manifest failed. exit_code=$PREP_EXIT"',
        '  echo "[HINT] On Windows, datasets/torchcodec requires FFmpeg full shared DLLs on PATH."',
        '  exit "$PREP_EXIT"',
        'done',
        '[ -s "$MANIFEST" ] || { echo "[ERROR] Manifest missing or empty: $MANIFEST"; exit 1; }',
        "",
        'echo "[2/2] Training..."',
        '"$UV" run --no-sync python "$IRODORI/train.py" \\',
        '  --config "$CONFIG" \\',
        '  --manifest "$MANIFEST" \\',
        '  --init-checkpoint "$CHECKPOINT" \\',
        '  --output-dir "$OUTPUT_DIR" \\',
        '  --device "$DEVICE" \\',
        '  --precision "$PRECISION" \\',
        '  --max-steps "$MAX_STEPS" \\',
        '  --batch-size "$BATCH_SIZE" \\',
        '  --gradient-accumulation-steps "$GRAD_ACCUM" \\',
        '  --speaker-inversion-tokens "$TOKENS" \\',
        '  --lr "$LR" \\',
        '  --save-every "$SAVE_EVERY" \\',
        '  --text-condition-dropout 0.0 \\',
        '  --speaker-condition-dropout 0.0 \\',
        '  --caption-condition-dropout 0.0 \\',
        '  --duration-speaker-dropout 0.0',
    ]
    if init_emb:
        sh[-1] += " \\"
        sh.append(f'  --speaker-inversion-init-embedding "{init_emb}"')
    sh.append("")
    sh.append('echo "Done. Embedding: $OUTPUT_DIR/checkpoint_final.speaker.safetensors"')
    sh_path = dataset_dir / "train.sh"
    sh_path.write_text("\n".join(sh), encoding="utf-8", newline="\n")
    try:
        import stat as _stat
        sh_path.chmod(sh_path.stat().st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)
    except Exception:
        pass


# ─── Prepare manifest (SSE) ───────────────────────────────────────────────────

class PrepareManifestRequest(BaseModel):
    job_name: str
    device: str = "auto"
    normalize_db: str = "-16.0"
    max_seconds: float = 0.0
    irodori_root: str = ""
    uv_exe: str = ""


@app.post("/api/prepare_manifest")
async def prepare_manifest(req: PrepareManifestRequest):
    irodori = _effective_irodori_root(req.irodori_root)
    uv = _effective_uv(req.uv_exe)
    dataset_dir = DATASETS_DIR / req.job_name
    source_jsonl = dataset_dir / "source.jsonl"
    manifest_path = dataset_dir / "manifest.jsonl"
    latent_dir = dataset_dir / "latents"

    if not source_jsonl.exists():
        raise HTTPException(400, "Run 'Build Dataset' first")

    requested_device = _normalise_device(req.device)
    primary_device = await _detect_device(uv, irodori) if requested_device == "auto" else requested_device
    log_path = _log_path("manifest", req.job_name)

    def build_cmd(device: str) -> list[str]:
        cmd = _uv_python(uv) + [
            str(WEBUI_DIR / "prepare_manifest_wrapper.py"),
            str(irodori / "prepare_manifest.py"),
            "--dataset", "json",
            "--data-files", f"train={source_jsonl}",
            "--split", "train",
            "--audio-column", "audio",
            "--text-column", "text",
            "--speaker-column", "speaker",
            "--speaker-id-prefix", req.job_name,
            "--output-manifest", str(manifest_path),
            "--latent-dir", str(latent_dir),
            "--device", device,
            "--normalize-db", req.normalize_db.strip() or "-16.0",
        ]
        if req.max_seconds and req.max_seconds > 0:
            cmd += ["--max-seconds", str(req.max_seconds)]
        return cmd

    async def stream_with_retry():
        for text in (
            f"[INFO] Log file: {log_path}",
            f"[INFO] Device: {primary_device}",
            "[INFO] If audio decoding fails on Windows, install/pass a full shared FFmpeg build on PATH.",
        ):
            _append_log_line(log_path, text)
            yield f"data: {json.dumps({'log': text})}\n\n"
        rc = 0
        async for payload in _sse_subprocess_iter(build_cmd(primary_device), cwd=str(irodori), log_path=log_path):
            event = json.loads(payload)
            if "rc" in event:
                rc = int(event["rc"])
            else:
                yield f"data: {json.dumps(event)}\n\n"
        if rc != 0 and primary_device != "cpu":
            text = f"[WARN] Manifest preparation failed on {primary_device}; retrying on cpu."
            _append_log_line(log_path, text)
            yield f"data: {json.dumps({'log': text})}\n\n"
            async for payload in _sse_subprocess_iter(build_cmd("cpu"), cwd=str(irodori), log_path=log_path):
                event = json.loads(payload)
                if "rc" in event:
                    rc = int(event["rc"])
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        if rc == 0 and (not manifest_path.exists() or manifest_path.stat().st_size <= 0):
            text = "[ERROR] Manifest missing or empty after preparation."
            _append_log_line(log_path, text)
            yield f"data: {json.dumps({'log': text})}\n\n"
            rc = 1
        yield f"data: {json.dumps({'done': True, 'rc': rc})}\n\n"

    return StreamingResponse(stream_with_retry(), media_type="text/event-stream")


# ─── Train (SSE) ──────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    job_name: str
    device: str = "auto"
    precision: str = "bf16"
    max_steps: int = 3000
    batch_size: int = 16
    grad_accum: int = 1
    tokens: int = 16
    learning_rate: float = 0.01
    save_every: int = 250
    init_embedding: str = ""
    checkpoint_path: str = ""
    config_path: str = ""
    irodori_root: str = ""
    uv_exe: str = ""


@app.post("/api/train")
async def train(req: TrainRequest):
    irodori = _effective_irodori_root(req.irodori_root)
    uv = _effective_uv(req.uv_exe)
    dataset_dir = DATASETS_DIR / req.job_name
    manifest_path = dataset_dir / "manifest.jsonl"
    output_dir = SPEAKER_OUTPUTS_DIR / req.job_name

    if not manifest_path.exists():
        raise HTTPException(400, "Run 'Prepare Manifest' first")

    saved_checkpoint = _settings.get("checkpoint_path", "").strip()
    checkpoint = Path(req.checkpoint_path or saved_checkpoint) if (req.checkpoint_path or saved_checkpoint).strip() else (irodori / "Irodori-TTS-500M-v3" / "model.safetensors")
    config = Path(req.config_path) if req.config_path.strip() else (irodori / "configs" / "train_500m_v3_speaker_inversion.yaml")
    requested_device = _normalise_device(req.device)
    train_device = await _detect_device(uv, irodori) if requested_device == "auto" else requested_device
    log_path = _log_path("train", req.job_name)

    cmd = _uv_run(uv, irodori, "train.py") + [
        "--config", str(config),
        "--manifest", str(manifest_path),
        "--init-checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--device", train_device,
        "--precision", req.precision,
        "--max-steps", str(req.max_steps),
        "--batch-size", str(req.batch_size),
        "--gradient-accumulation-steps", str(req.grad_accum),
        "--speaker-inversion-tokens", str(req.tokens),
        "--lr", str(req.learning_rate),
        "--save-every", str(req.save_every),
        "--text-condition-dropout", "0.0",
        "--speaker-condition-dropout", "0.0",
        "--caption-condition-dropout", "0.0",
        "--duration-speaker-dropout", "0.0",
    ]
    if req.init_embedding.strip():
        cmd += ["--speaker-inversion-init-embedding", req.init_embedding.strip()]

    async def stream():
        for text in (f"[INFO] Log file: {log_path}", f"[INFO] Device: {train_device}"):
            _append_log_line(log_path, text)
            yield f"data: {json.dumps({'log': text})}\n\n"
        async for payload in _sse_subprocess(cmd, cwd=str(irodori), log_path=log_path):
            yield payload

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─── Generate (SSE) ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str
    embedding_path: str
    checkpoint_path: str = ""
    output_name: str = "output"
    num_steps: int = 40
    seed: int = -1
    irodori_root: str = ""
    uv_exe: str = ""


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    irodori = _effective_irodori_root(req.irodori_root)
    uv = _effective_uv(req.uv_exe)
    output_wav = OUTPUTS_DIR / f"{req.output_name}.wav"
    saved_checkpoint = _settings.get("checkpoint_path", "").strip()
    checkpoint = Path(req.checkpoint_path or saved_checkpoint) if (req.checkpoint_path or saved_checkpoint).strip() else (irodori / "Irodori-TTS-500M-v3" / "model.safetensors")
    log_path = _log_path("generate", req.output_name)
    request_path = log_path.with_suffix(".request.json")
    request_path.write_text(
        json.dumps({
            "checkpoint": str(checkpoint),
            "embedding_path": req.embedding_path,
            "text": req.text,
            "output_wav": str(output_wav),
            "num_steps": req.num_steps,
            "seed": req.seed,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cmd = _uv_python(uv) + [
        str(WEBUI_DIR / "infer_worker.py"),
        str(irodori / "infer.py"),
        str(request_path),
    ]

    async def stream():
        for text in (
            f"[INFO] Log file: {log_path}",
            f"[INFO] Request file: {request_path}",
            f"[INFO] Text: {req.text}",
        ):
            _append_log_line(log_path, text)
            yield f"data: {json.dumps({'log': text})}\n\n"
        async for chunk in _sse_subprocess(cmd, cwd=str(irodori), log_path=log_path):
            yield chunk
        if output_wav.exists():
            yield f"data: {json.dumps({'audio_url': f'/audio/output/{output_wav.name}'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─── SSE subprocess helper ────────────────────────────────────────────────────

# Known Windows NTSTATUS / CRT exit codes that appear as large positive integers.
_WIN_RC_HINTS: dict[int, str] = {
    0xC0000005: "ACCESS_VIOLATION – native code crash (GPU driver / CUDA / DLL issue). "
                "Try: set Device to 'cpu', or check GPU driver compatibility.",
    0xC000013A: "STATUS_CONTROL_C_EXIT – process was interrupted (Ctrl+C).",
    0xC00000FD: "STACK_OVERFLOW – recursion or stack exhausted in native code.",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN – security check failed in native code.",
    0xC0000374: "HEAP_CORRUPTION – memory corruption in native extension.",
    0xE0434352: "CLR (.NET) unhandled exception propagated to process.",
    3:           "Fatal Python error / early crash (check stderr output).",
}


def _rc_hint(rc: int) -> str:
    """Return a human-readable hint for a non-zero exit code."""
    if rc == 0:
        return ""
    # Normalise to unsigned 32-bit (Windows returns negative ints as signed)
    urc = rc & 0xFFFFFFFF
    hint = _WIN_RC_HINTS.get(urc) or _WIN_RC_HINTS.get(rc)
    hex_str = f"0x{urc:08X}"
    if hint:
        return f"Exit code {rc} ({hex_str}): {hint}"
    return f"Exit code {rc} ({hex_str})"


async def _sse_subprocess_iter(cmd: list[str], cwd: str | None = None, log_path: Path | None = None):
    log_f = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")

    def write_log(text: str) -> None:
        if log_f:
            log_f.write(text + "\n")
            log_f.flush()

    display_cmd = " ".join(
        f'"{p}"' if " " in str(p) else str(p) for p in cmd
    )
    write_log("$ " + display_cmd)
    yield json.dumps({"log": "$ " + display_cmd})

    try:
        env = _subprocess_env(Path(cwd) if cwd else IRODORI_ROOT)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        assert proc.stdout is not None
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            write_log(text)
            yield json.dumps({"log": text})
        rc = await proc.wait()

        hint = _rc_hint(rc)
        if hint:
            write_log(hint)
            yield json.dumps({"log": hint})
        write_log(f"[Done rc={rc}]")
        yield json.dumps({"rc": rc})
    finally:
        if log_f:
            log_f.close()


async def _sse_subprocess(cmd: list[str], cwd: str | None = None, log_path: Path | None = None):
    async for payload in _sse_subprocess_iter(cmd, cwd=cwd, log_path=log_path):
        event = json.loads(payload)
        if "rc" in event:
            yield f"data: {json.dumps({'done': True, 'rc': event['rc']})}\n\n"
        else:
            yield f"data: {json.dumps(event)}\n\n"


# ─── Entry point ──────────────────────────────────────────────────────────────

def _find_free_port(host: str, start: int, tries: int = 20) -> int:
    import socket
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + tries - 1}")


def main() -> None:
    global IRODORI_ROOT
    import argparse
    parser = argparse.ArgumentParser(description="Speaker Inversion Builder WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--irodori-root", default=str(IRODORI_ROOT))
    args = parser.parse_args()

    IRODORI_ROOT = Path(args.irodori_root)

    port = _find_free_port(args.host, args.port)
    if port != args.port:
        print(f"[INFO] Port {args.port} in use, using {port} instead")

    import socket as _sock
    hostname = _sock.gethostname()
    try:
        local_ip = _sock.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n  Speaker Inversion Builder")
    print(f"  Local:   http://127.0.0.1:{port}")
    if args.host == "0.0.0.0":
        print(f"  Network: http://{local_ip}:{port}")
    print()

    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
