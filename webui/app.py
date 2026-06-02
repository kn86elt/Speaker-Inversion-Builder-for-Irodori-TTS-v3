#!/usr/bin/env python3
"""Speaker Inversion WebUI – FastAPI backend."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
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
STATE_FILE = DATA_DIR / "state.json"

for _d in (DATA_DIR, UPLOADS_DIR, SEGMENTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# irodori-TTS root – overridable via --irodori-root CLI arg
IRODORI_ROOT = Path("C:/usr/sd/Irodori-TTS-v3")

# ─── State ────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()


def _empty_state() -> dict:
    return {"files": {}, "segments": []}


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


# ─── Whisper ──────────────────────────────────────────────────────────────────
_whisper_model: Any = None
_whisper_lock = threading.Lock()


def _get_whisper(model_size: str = "medium"):
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute)
    return _whisper_model


def _do_transcribe(wav_path: Path) -> str:
    model = _get_whisper()
    segments, _ = model.transcribe(str(wav_path), language="ja", beam_size=5)
    return "".join(seg.text for seg in segments).strip()


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
    p = UPLOADS_DIR / f"{file_id}.wav"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")


@app.get("/audio/segment/{seg_id}")
async def serve_segment(seg_id: str):
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

        dur = await loop.run_in_executor(_thread_pool, _get_duration, wav)

        with _state_lock:
            _state["files"][file_id] = {
                "id": file_id,
                "name": file.filename,
                "duration": round(dur, 3),
            }
            _save_state(_state)

        added.append(file_id)

    return {"added": added}


# ─── File operations ──────────────────────────────────────────────────────────

@app.delete("/api/file/{file_id}")
async def delete_file(file_id: str):
    with _state_lock:
        del_segs = [s["id"] for s in _state["segments"] if s["file_id"] == file_id]
        _state["segments"] = [s for s in _state["segments"] if s["file_id"] != file_id]
        _state["files"].pop(file_id, None)
        _save_state(_state)
    for sid in del_segs:
        (SEGMENTS_DIR / f"{sid}.wav").unlink(missing_ok=True)
    (UPLOADS_DIR / f"{file_id}.wav").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/file/{file_id}/as_segment")
async def file_as_segment(file_id: str):
    wav = UPLOADS_DIR / f"{file_id}.wav"
    if not wav.exists():
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


# ─── Split on silence ─────────────────────────────────────────────────────────

class SilenceParams(BaseModel):
    min_silence_ms: int = 700
    silence_thresh_db: float = -40.0
    keep_silence_ms: int = 200


@app.post("/api/file/{file_id}/split_silence")
async def split_file_silence(file_id: str, params: SilenceParams):
    wav = UPLOADS_DIR / f"{file_id}.wav"
    if not wav.exists():
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
    src = SEGMENTS_DIR / f"{seg_id}.wav"

    seg_data: list[tuple[str, float, float]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        new_id = str(uuid.uuid4())
        dst = SEGMENTS_DIR / f"{new_id}.wav"
        await loop.run_in_executor(_thread_pool, _extract_segment, src, s, e, dst)
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
    p = SEGMENTS_DIR / f"{seg_id}.wav"
    if not p.exists():
        raise HTTPException(404)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_thread_pool, _do_transcribe, p)
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
        for i, seg in enumerate(pending):
            p = SEGMENTS_DIR / f"{seg['id']}.wav"
            if not p.exists():
                yield f"data: {json.dumps({'id': seg['id'], 'error': 'file missing', 'progress': i + 1, 'total': total})}\n\n"
                continue
            try:
                text = await loop.run_in_executor(_thread_pool, _do_transcribe, p)
                with _state_lock:
                    s = next((x for x in _state["segments"] if x["id"] == seg["id"]), None)
                    if s:
                        s["text"] = text
                        s["transcribed"] = True
                    _save_state(_state)
                yield f"data: {json.dumps({'id': seg['id'], 'text': text, 'progress': i + 1, 'total': total})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'id': seg['id'], 'error': str(exc), 'progress': i + 1, 'total': total})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

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
        _save_state(_state)
    return {"ok": True}


@app.delete("/api/segment/{seg_id}")
async def delete_segment(seg_id: str):
    with _state_lock:
        _state["segments"] = [s for s in _state["segments"] if s["id"] != seg_id]
        _save_state(_state)
    (SEGMENTS_DIR / f"{seg_id}.wav").unlink(missing_ok=True)
    return {"ok": True}


# ─── Dataset build ────────────────────────────────────────────────────────────

class BuildDatasetRequest(BaseModel):
    job_name: str
    speaker_name: str = ""


@app.post("/api/build_dataset")
async def build_dataset(req: BuildDatasetRequest):
    with _state_lock:
        segments = [s for s in _state["segments"] if s.get("text", "").strip()]

    if not segments:
        raise HTTPException(400, "No segments with text found")

    job_dir = DATA_DIR / "runs" / req.job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    speaker = req.speaker_name.strip() or req.job_name

    jsonl_path = job_dir / "source_dataset.jsonl"
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            seg_wav = SEGMENTS_DIR / f"{seg['id']}.wav"
            if seg_wav.exists():
                f.write(json.dumps({
                    "audio": str(seg_wav.resolve()),
                    "text": seg["text"],
                    "speaker": speaker,
                }, ensure_ascii=False) + "\n")
                count += 1

    return {"path": str(jsonl_path), "count": count}


# ─── Runs list ────────────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs():
    runs_dir = DATA_DIR / "runs"
    if not runs_dir.exists():
        return {"runs": []}
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        embed = d / "checkpoints" / "checkpoint_final.speaker.safetensors"
        runs.append({
            "name": d.name,
            "has_embedding": embed.exists(),
            "embedding_path": str(embed) if embed.exists() else None,
        })
    return {"runs": runs}


# ─── Settings ────────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    irodori = IRODORI_ROOT
    return {
        "irodori_root": str(irodori),
        "default_checkpoint": str(irodori / "Irodori-TTS-500M-v3" / "model.safetensors"),
        "default_config": str(irodori / "configs" / "train_500m_v3_speaker_inversion.yaml"),
        "python_exe": sys.executable,
    }


# ─── Prepare manifest (SSE) ───────────────────────────────────────────────────

class PrepareManifestRequest(BaseModel):
    job_name: str
    device: str = "cuda"
    normalize_db: str = "-16.0"
    max_seconds: float = 0.0
    irodori_root: str = ""
    python_exe: str = ""


@app.post("/api/prepare_manifest")
async def prepare_manifest(req: PrepareManifestRequest):
    irodori = Path(req.irodori_root) if req.irodori_root.strip() else IRODORI_ROOT
    python = req.python_exe.strip() or sys.executable
    job_dir = DATA_DIR / "runs" / req.job_name
    source_jsonl = job_dir / "source_dataset.jsonl"
    manifest_path = job_dir / "manifest.jsonl"
    latent_dir = job_dir / "latents"

    if not source_jsonl.exists():
        raise HTTPException(400, "Run 'Build Dataset' first")

    cmd = [
        python, str(irodori / "prepare_manifest.py"),
        "--dataset", "json",
        "--data-files", f"train={source_jsonl}",
        "--split", "train",
        "--audio-column", "audio",
        "--text-column", "text",
        "--speaker-column", "speaker",
        "--speaker-id-prefix", req.job_name,
        "--output-manifest", str(manifest_path),
        "--latent-dir", str(latent_dir),
        "--device", req.device,
        "--normalize-db", req.normalize_db.strip() or "-16.0",
    ]
    if req.max_seconds and req.max_seconds > 0:
        cmd += ["--max-seconds", str(req.max_seconds)]

    return StreamingResponse(_sse_subprocess(cmd, cwd=str(irodori)), media_type="text/event-stream")


# ─── Train (SSE) ──────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    job_name: str
    device: str = "cuda"
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
    python_exe: str = ""


@app.post("/api/train")
async def train(req: TrainRequest):
    irodori = Path(req.irodori_root) if req.irodori_root.strip() else IRODORI_ROOT
    python = req.python_exe.strip() or sys.executable
    job_dir = DATA_DIR / "runs" / req.job_name
    manifest_path = job_dir / "manifest.jsonl"
    output_dir = job_dir / "checkpoints"

    if not manifest_path.exists():
        raise HTTPException(400, "Run 'Prepare Manifest' first")

    checkpoint = Path(req.checkpoint_path) if req.checkpoint_path.strip() else (irodori / "Irodori-TTS-500M-v3" / "model.safetensors")
    config = Path(req.config_path) if req.config_path.strip() else (irodori / "configs" / "train_500m_v3_speaker_inversion.yaml")

    cmd = [
        python, str(irodori / "train.py"),
        "--config", str(config),
        "--manifest", str(manifest_path),
        "--init-checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--device", req.device,
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

    return StreamingResponse(_sse_subprocess(cmd, cwd=str(irodori)), media_type="text/event-stream")


# ─── Generate (SSE) ──────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str
    embedding_path: str
    checkpoint_path: str = ""
    output_name: str = "output"
    num_steps: int = 40
    seed: int = -1
    irodori_root: str = ""
    python_exe: str = ""


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    irodori = Path(req.irodori_root) if req.irodori_root.strip() else IRODORI_ROOT
    python = req.python_exe.strip() or sys.executable
    output_wav = OUTPUTS_DIR / f"{req.output_name}.wav"
    checkpoint = Path(req.checkpoint_path) if req.checkpoint_path.strip() else (irodori / "Irodori-TTS-500M-v3" / "model.safetensors")

    cmd = [
        python, str(irodori / "infer.py"),
        "--checkpoint", str(checkpoint),
        "--ref-embed", req.embedding_path,
        "--text", req.text,
        "--output-wav", str(output_wav),
        "--num-steps", str(req.num_steps),
    ]
    if req.seed >= 0:
        cmd += ["--seed", str(req.seed)]

    async def stream():
        async for chunk in _sse_subprocess(cmd, cwd=str(irodori)):
            yield chunk
        if output_wav.exists():
            yield f"data: {json.dumps({'audio_url': f'/audio/output/{output_wav.name}'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─── SSE subprocess helper ────────────────────────────────────────────────────

async def _sse_subprocess(cmd: list[str], cwd: str | None = None):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        text = line.decode("utf-8", errors="replace").rstrip()
        yield f"data: {json.dumps({'log': text})}\n\n"
    rc = await proc.wait()
    yield f"data: {json.dumps({'done': True, 'rc': rc})}\n\n"


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
