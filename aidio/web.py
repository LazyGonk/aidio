"""FastAPI web interface for Aidio.

Two capabilities, both optional-LLM:
- POST /analyze: LLM audio-characteristics analysis (upload → analysis report)
- POST /transcribe: Whisper lyrics transcription as a background job
  (job id → GET /jobs/{id}), plus POST /transcribe/embed which returns
  the original file with an embedded ID3v2 USLT lyrics tag.
- GET/POST /settings: LLM endpoint/api-key/model configuration overrides.
- POST /explain: LLM analysis of (transcribed) lyrics text.
- POST /correct: LLM polish of editable lyrics text (on demand).
- POST /jobs/{id}/cancel: cooperative cancellation of a running job.
- POST /bundle/zip: zip many finished result texts into one download.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aidio import __copyright__, __version__
from aidio.analyzer import analyze_audio
from aidio.lyrics.cuda_dlls import enable_cuda_dlls
from aidio.lyrics.device import DeviceError, resolve_device

app = FastAPI(title="Aidio - Audio Analysis Engine")

# In-memory job registry (single-user local use; jobs are ephemeral).
_jobs: dict[str, dict] = {}

_FINISHED_STATUSES = ("done", "error", "cancelled")


class _JobCancelled(Exception):
    """Raised from the progress callback to abort a cancelled job cooperatively."""


@dataclass
class WebSettings:
    """Effective LLM settings for one request.

    Precedence: browser cookie ("aidio_settings") > env defaults. Settings
    are NOT persisted server-side — the cookie is the user's choice; env
    (.env) is the deployment default.
    """

    endpoint: str = ""
    api_key: str = ""
    llm_model: str = ""

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint or os.environ.get(
            "OPENAI_BASE_URL", "http://127.0.0.1:4000/v1"
        )
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "sk-not-needed-for-local")
        self.llm_model = self.llm_model or os.environ.get("ANALYSIS_MODEL", "gemma4:12b")


SETTINGS_COOKIE = "aidio_settings"


def _effective_settings(request: Request) -> WebSettings:
    """Env defaults + optional per-request overrides from the settings cookie."""
    override = WebSettings()
    raw = request.cookies.get(SETTINGS_COOKIE)
    if raw:
        try:
            data = json.loads(base64.b64decode(raw))
            if isinstance(data, dict):
                override.endpoint = str(data.get("endpoint") or "") or override.endpoint
                override.api_key = str(data.get("api_key") or "") or override.api_key
                override.llm_model = str(data.get("llm_model") or "") or override.llm_model
        except (ValueError, json.JSONDecodeError):
            pass  # malformed cookie: fall back to env
    return override


class SettingsPayload(BaseModel):
    """POST /settings body; empty/missing fields keep current values."""

    endpoint: str | None = None
    api_key: str | None = None
    llm_model: str | None = None


class ExplainPayload(BaseModel):
    """POST /explain body: lyrics text plus optional LLM overrides."""

    lyrics: str
    endpoint: str | None = None
    model: str | None = None
    api_key: str | None = None


class CorrectPayload(BaseModel):
    """POST /correct body: lyrics text plus optional correction notes."""

    lyrics: str
    notes: str | None = None
    endpoint: str | None = None
    model: str | None = None
    api_key: str | None = None


class BundleFile(BaseModel):
    """One text file inside a downloadable zip bundle."""

    name: str
    text: str


class BundlePayload(BaseModel):
    """POST /bundle/zip body: finished result texts from the browser."""

    files: list[BundleFile]


# Serve static files (CSS, JS)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main upload page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return _render_index(html_path.read_text(encoding="utf-8"))
    return _default_html()


def _render_index(html: str) -> str:
    """Fill template placeholders with package metadata (version, copyright)."""
    return html.replace("__AIDIO_VERSION__", __version__).replace(
        "__AIDIO_COPYRIGHT__", __copyright__
    )


@app.post("/analyze")
async def analyze(
    request: Request,
    audio_file: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    """Analyze an uploaded audio file.

    Parameters
    ----------
    audio_file : UploadFile
        The audio file to analyze (WAV, MP3, FLAC, etc.)
    prompt : str
        Custom analysis prompt (optional)

    Returns
    -------
    dict
        Analysis result with text and metadata
    """
    eff = _effective_settings(request)
    # Save the upload under its real suffix (format sniffing is not the contract).
    filename = Path(audio_file.filename or "audio.wav").name
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run analysis (LLM settings from browser cookie or env)
        from aidio.client import AnalysisClient

        client = AnalysisClient(
            base_url=eff.endpoint,
            api_key=eff.api_key,
            model=eff.llm_model,
        )
        try:
            result = analyze_audio(tmp_path, prompt=prompt or None, client=client)
        finally:
            client.close()

        return {
            "success": True,
            "chunk_count": result.chunk_count,
            "total_duration": round(result.total_duration_seconds, 2),
            "analysis": result.full_text,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/device")
async def get_device() -> dict:
    """Current default device (env AIDIO_DEVICE or cuda) for the UI dropdown."""
    return {"device": _default_device()}


@app.get("/settings")
async def get_settings(request: Request) -> dict:
    """Env-default LLM settings (api_key masked).

    The browser may keep overrides in its own cookie ("aidio_settings");
    the server never persists them. Use /settings/effective to see what a
    request would actually use.
    """
    s = WebSettings()
    masked = "•••" if s.api_key else ""
    return {
        "endpoint": s.endpoint,
        "api_key": masked,
        "has_api_key": bool(s.api_key),
        "llm_model": s.llm_model,
        "persist": "cookie",
    }


@app.post("/settings")
async def post_settings(payload: SettingsPayload, response: Response) -> dict:
    """Validate submitted settings and hand them back as a Set-Cookie header.

    The server does not persist settings — the browser cookie is the
    user's choice. Subsequent requests override env defaults via the
    "aidio_settings" cookie for LLM endpoint/api key/model (audio
    analysis, lyric correction, lyric analysis). Change .env instead for
    deployment-level defaults.
    """
    value: dict = {}
    if payload.endpoint is not None and payload.endpoint.strip():
        value["endpoint"] = payload.endpoint.strip()
    if payload.api_key is not None and payload.api_key:
        value["api_key"] = payload.api_key
    if payload.llm_model is not None and payload.llm_model.strip():
        value["llm_model"] = payload.llm_model.strip()
    response.set_cookie(
        SETTINGS_COOKIE,
        base64.b64encode(json.dumps(value).encode()).decode(),
        samesite="strict",
    )
    return {
        "success": True,
        "note": "Stored in browser cookie (aidio_settings); .env stays default.",
        **value,
    }


@app.get("/settings/effective")
async def get_settings_effective(request: Request) -> dict:
    """Effective settings for the current browser (api_key masked)."""
    eff = _effective_settings(request)
    masked = "•••" if eff.api_key else ""
    return {
        "endpoint": eff.endpoint,
        "api_key": masked,
        "has_api_key": bool(eff.api_key),
        "llm_model": eff.llm_model,
        "validated": True,
    }


@app.post("/explain")
async def post_explain(request: Request, payload: ExplainPayload) -> dict:
    """LLM analysis of lyrics text (e.g. Whisper transcription output).

    Synchronous; usually fast because lyrics are small. Returns structured
    analysis: title, themes, motifs, characters, narrative arc.
    """
    eff = _effective_settings(request)
    from aidio.lyrics.explain import explain_lyrics  # deferred: heavy

    result = explain_lyrics(
        payload.lyrics,
        endpoint=payload.endpoint or eff.endpoint,
        model=payload.model or eff.llm_model,
        api_key=payload.api_key or eff.api_key,
    )
    return {
        "success": True,
        "title": result.title,
        "themes": result.themes,
        "motifs": result.motifs,
        "characters": result.characters,
        "narrative_arc": result.narrative_arc,
        "worldbuilding_notes": result.worldbuilding_notes,
        "suspicious_lines": result.suspicious_lines,
        "tone": result.tone,
    }


@app.post("/correct")
async def post_correct(request: Request, payload: CorrectPayload) -> dict:
    """Polish lyrics on demand (the result card's *Polish lyrics* button).

    Uses the built-in correction prompt (or AIDIO_CORRECTION_PROMPT) and
    the browser's effective LLM settings. ``notes`` is appended as extra
    user instructions, e.g. artist names or spelling preferences.
    """
    eff = _effective_settings(request)
    from aidio.lyrics.correct import correct_lyrics  # deferred: optional heavy dep

    corrected = correct_lyrics(
        payload.lyrics,
        context=payload.notes or None,
        endpoint=payload.endpoint or eff.endpoint,
        model=payload.model or eff.llm_model,
        api_key=payload.api_key or eff.api_key,
    )
    return {
        "success": True,
        "lyrics": corrected,
        "changed": corrected != payload.lyrics,
    }


def _validate_web_device(device: str | None) -> str:
    """Mirror device.py validation, returning HTTP-400-friendly errors."""
    try:
        return resolve_device(device)
    except DeviceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _default_device() -> str:
    """Client-facing default device: env override, else best available.

    Detection is best-effort so the UI opens on CPU instead of failing on
    machines without a CUDA-capable GPU.
    """
    env_device = os.environ.get("AIDIO_DEVICE")
    if env_device:
        try:
            return resolve_device(env_device)
        except DeviceError:
            pass

    try:
        enable_cuda_dlls()
        import ctranslate2  # noqa: PLC0415 - optional heavy dependency

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        pass

    try:
        import torch  # noqa: PLC0415 - optional heavy dependency

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    try:
        return resolve_device(None)
    except DeviceError:
        return "cuda"


@app.post("/transcribe")
async def post_transcribe(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    backend: str = Form(default="local"),
    device: str = Form(default=""),
    model: str = Form(default="large-v3-turbo"),
    language: str = Form(default="en"),
    separate_vocals: bool = Form(default=False),
    timestamps: bool = Form(default=False),
    llm_correct: bool = Form(default=False),
    correction_context: str = Form(default=""),
) -> dict:
    """Start a lyrics-transcription job; returns a job id to poll later.

    backend: "local" (faster-whisper on device) or "remote" (Whisper API,
    configured via WHISPER_URL/WHISPER_MODEL env). The file is written to
    a temp dir preserving its original extension. Transcription takes
    minutes, so work runs in a background task.
    """
    backend = backend if backend in ("local", "remote") else "local"
    resolved = _validate_web_device(device or None)
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        Path(tmp.name).write_bytes(await audio_file.read())
        tmp_path = tmp.name

    job_id = uuid.uuid4().hex
    original_name = Path(audio_file.filename or "song.mp3").name
    eff = _effective_settings(request)
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "stage": "Queued",
        "device": resolved,
        "original_name": original_name,
        "cancel_requested": False,
    }
    context = correction_context or None

    def _run() -> None:
        job = _jobs[job_id]

        def _check_cancel() -> None:
            if job.get("cancel_requested"):
                raise _JobCancelled

        job["status"] = "running"
        job["stage"] = "Starting"

        def _progress(fraction: float, stage: str) -> None:
            _check_cancel()
            job["progress"] = round(float(fraction) * 100, 1)
            job["stage"] = stage

        try:
            _check_cancel()
            from aidio.pipeline import transcribe_lyrics  # deferred: heavy

            result = transcribe_lyrics(
                tmp_path,
                backend=backend,
                device=resolved,
                model=model,
                language=language or None,
                separate_vocals=separate_vocals,
                timestamps=timestamps,
                llm_correct=llm_correct,
                correction_context=context,
                endpoint=eff.endpoint,
                model_name=eff.llm_model,
                progress=_progress,
            )
            _check_cancel()
            job["progress"] = 100.0
            job["stage"] = "Done"
            job["status"] = "done"
            job["result"] = {
                "lyrics": result.lyrics,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
                ],
                "device": result.device,
                "backend": backend,
                "used_demucs": result.used_demucs,
                "corrections": result.corrections,
            }
        except _JobCancelled:
            job["status"] = "cancelled"
            job["stage"] = "Cancelled"
            job["progress"] = 0.0
        except Exception as e:
            job["status"] = "error"
            job["stage"] = "Failed"
            job["error"] = str(e)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    background_tasks.add_task(_run)
    return {"success": True, "job_id": job_id, "device": resolved}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """Poll a transcription job by id."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return {"success": True, **job}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Request cancellation of a pending or running transcription job.

    Cancellation is cooperative: the worker stops at the next progress
    report (Whisper yields segments frequently) and the job becomes
    "cancelled". Finished jobs cannot be cancelled.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.get("status") in _FINISHED_STATUSES:
        raise HTTPException(status_code=409, detail="Job already finished")
    job["cancel_requested"] = True
    job["status"] = "cancelling"
    job["stage"] = "Cancelling"
    return {"success": True, "job_id": job_id, "status": job["status"]}


@app.get("/jobs/{job_id}/lyrics.txt")
async def get_job_lyrics_txt(background_tasks: BackgroundTasks, job_id: str) -> FileResponse:
    """Download the finished lyrics as <original stem>.txt.

    Connects to: aidio/lyrics/output.py save_lyrics_txt — same naming
    convention '<mp3 stem>.txt' as the CLI, but served to the browser.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=409, detail="Job not finished")

    filename = Path(job.get("original_name", "lyrics")).stem + ".txt"
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", encoding="utf-8", newline="\n", suffix=".txt"
    ) as tmp:
        tmp.write(result["lyrics"])  # newline="\n" preserves LF
        path = Path(tmp.name)

    background_tasks.add_task(Path(path).unlink, True)
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=filename,
    )


def _safe_bundle_name(name: str, fallback: str) -> str:
    """Strip directories and characters that are unsafe inside a zip entry."""
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return base or fallback


@app.post("/bundle/zip")
async def bundle_zip(background_tasks: BackgroundTasks, payload: BundlePayload) -> FileResponse:
    """Zip the browser's finished result texts into one download.

    The browser sends the current (possibly edited) text for each result,
    so this works for both Whisper transcriptions and LLM analyses.
    """
    if not payload.files:
        raise HTTPException(status_code=400, detail="No files to bundle")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    archive = Path(tmp.name)
    used: set[str] = set()
    try:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for index, item in enumerate(payload.files, start=1):
                name = _safe_bundle_name(item.name, f"result-{index}.txt")
                stem = Path(name).stem
                suffix = Path(name).suffix or ".txt"
                candidate = f"{stem}{suffix}"
                counter = 2
                while candidate.lower() in used:
                    candidate = f"{stem}-{counter}{suffix}"
                    counter += 1
                used.add(candidate.lower())
                zf.writestr(candidate, item.text)
    except Exception:
        archive.unlink(missing_ok=True)
        raise

    background_tasks.add_task(Path(archive).unlink, True)
    return FileResponse(archive, media_type="application/zip", filename="aidio-lyrics.zip")


@app.post("/embed")
async def post_embed(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    lyrics: str = Form(...),
) -> FileResponse:
    """Embed already-transcribed (and possibly edited) lyrics into audio.

    Unlike /transcribe/embed, this never runs transcription: the browser
    sends the text currently shown in the editable result box.
    """
    filename = Path(audio_file.filename or "audio.mp3").name
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        Path(tmp.name).write_bytes(await audio_file.read())
        src = Path(tmp.name)

    try:
        from aidio.lyrics.output import embed_lyrics  # deferred: mutagen

        embed_lyrics(lyrics, src)
    except Exception as e:
        src.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e

    background_tasks.add_task(Path(src).unlink, True)
    return FileResponse(
        src,
        media_type="application/octet-stream",
        filename=f"{Path(filename).stem}-lyrics{suffix}",
        headers={"X-Lyrics-Done": "1"},
    )


@app.post("/transcribe/embed")
async def post_transcribe_embed(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    backend: str = Form(default="local"),
    device: str = Form(default=""),
    model: str = Form(default="large-v3-turbo"),
    language: str = Form(default="en"),
    separate_vocals: bool = Form(default=False),
    llm_correct: bool = Form(default=False),
) -> FileResponse:
    """Transcribe lyrics synchronously and return the file with a USLT tag.

    Blocking endpoint — the original bytes (not a re-encoded temp WAV) are
    returned with the ID3v2 lyrics tag embedded, as a download. The temp
    copy is deleted after the response is sent.
    """
    eff = _effective_settings(request)
    backend = backend if backend in ("local", "remote") else "local"
    resolved = _validate_web_device(device or None)
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        Path(tmp.name).write_bytes(await audio_file.read())
        src = Path(tmp.name)

    from aidio.pipeline import transcribe_lyrics  # deferred: heavy

    try:
        transcribe_lyrics(
            str(src),
            backend=backend,
            device=resolved,
            model=model,
            language=language or None,
            separate_vocals=separate_vocals,
            llm_correct=llm_correct,
            embed_into=src,
            endpoint=eff.endpoint,
            model_name=eff.llm_model,
        )
    except DeviceError as e:
        Path(src).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e

    background_tasks.add_task(Path(src).unlink, True)
    filename = Path(audio_file.filename or "audio.mp3").name
    return FileResponse(
        src,
        media_type="application/octet-stream",
        filename=f"{Path(filename).stem}-lyrics{Path(filename).suffix or '.mp3'}",
        headers={"X-Lyrics-Done": "1"},
    )


@app.get("/notices")
async def get_notices() -> HTMLResponse:
    """Third-party license notices (THIRD-PARTY-NOTICES.md) for the UI viewer."""
    path = Path(__file__).resolve().parent / "THIRD-PARTY-NOTICES.md"
    if not path.exists():
        return HTMLResponse(
            "<h1>Third-Party Notices</h1><p>No THIRD-PARTY-NOTICES.md found — "
            "generate with <code>python -m scripts.generate_notices</code>.</p>",
            status_code=404,
        )
    import html as _html

    text = _html.escape(path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Aidio - Third-Party Notices</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;"
        "margin:40px auto;padding:0 20px;}table{border-collapse:collapse;}"
        "td,th{border:1px solid #ccc;padding:4px 10px;}</style></head><body>"
        f"<pre style='font-size:0.95em'>{text}</pre></body></html>"
    )


def _default_html() -> str:
    """Fallback HTML if template file doesn't exist."""
    return _render_index("""
<!DOCTYPE html>
<html>
<head>
    <title>Aidio - Audio Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px;
               margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        button { background: #007bff; color: white; border: none; padding: 12px 24px;
                 cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        #result { margin-top: 20px; padding: 15px; background: #f8f9fa;
                  border-radius: 5px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Aidio - Audio Analysis Engine</h1>
    <form id="uploadForm" enctype="multipart/form-data">
        <div class="upload-area">
            <input type="file" name="audio_file" accept="audio/*" required>
        </div>
        <textarea name="prompt" rows="3" placeholder="Custom analysis prompt (optional)"></textarea>
        <br>
        <button type="submit">Analyze Audio</button>
    </form>
    <div id="result"></div>
    <p style="color:#666;font-size:0.85em;margin-top:30px">
        Aidio v__AIDIO_VERSION__ · __AIDIO_COPYRIGHT__ · MIT License
    </p>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = 'Analyzing...';

            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.success) {
                    resultDiv.textContent =
                        `Analyzed ${data.chunk_count} chunk(s), ${data.total_duration}s total\n\n` +
                        data.analysis;
                } else {
                    resultDiv.textContent = `Error: ${data.error}`;
                }
            } catch (err) {
                resultDiv.textContent = `Error: ${err.message}`;
            }
        });
    </script>
</body>
</html>
""")
