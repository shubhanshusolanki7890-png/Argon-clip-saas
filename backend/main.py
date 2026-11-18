import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    ALLOWED_FREE_GEMINI_MODELS,
    CLIPS_DIR,
    CORS_ORIGINS,
    FRONTEND_DIR,
    RuntimeConfig,
    SUPPORTED_VIDEO_EXTENSIONS,
    TEMP_DIR,
)
from backend.models import (
    ApiKeyConfigRequest,
    ClipRenderItem,
    JobStatus,
    RenderClipsRequest,
    YouTubeIngestRequest,
)
from backend.services.clipper import ClipGenerationError, render_vertical_clip
from backend.services.gemini_service import (
    GeminiServiceError,
    detect_best_moments_with_gemini,
    verify_gemini_connectivity,
)
from backend.services.job_store import (
    create_job,
    get_job,
    list_jobs,
    save_job,
    update_job_progress,
)
from backend.services.transcription import transcribe_video_audio
from backend.services.video_input import (
    VideoInputError,
    ingest_uploaded_file,
    ingest_youtube_url,
)

app = FastAPI(
    title="ArgonClip AI Clipping SaaS API",
    description="End-to-end Long-Form Video to 9:16 Vertical Short Clips with Free-Tier Gemini Analysis, Multilingual Transcription (Hindi/English/Hinglish), Subject Tracking, and Animated Word Captions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_binary(name: str) -> bool:
    return shutil.which(name) is not None


@app.get("/api/health")
def health_check():
    ffmpeg_ok = _check_binary("ffmpeg")
    ffprobe_ok = _check_binary("ffprobe")
    yt_dlp_installed = _check_binary("yt-dlp")
    if not yt_dlp_installed:
        try:
            import yt_dlp  # type: ignore  # noqa: F401
            yt_dlp_installed = True
        except ImportError:
            yt_dlp_installed = False

    return {
        "status": "healthy" if (ffmpeg_ok and ffprobe_ok) else "degraded",
        "ffmpeg_available": ffmpeg_ok,
        "ffprobe_available": ffprobe_ok,
        "yt_dlp_available": yt_dlp_installed,
        "gemini_api_key_configured": RuntimeConfig.has_api_key(),
        "active_gemini_model": RuntimeConfig.get_model(),
        "allowed_free_gemini_models": ALLOWED_FREE_GEMINI_MODELS,
        "supported_video_formats": sorted(SUPPORTED_VIDEO_EXTENSIONS),
    }


@app.get("/api/config/status")
def get_config_status():
    return {
        "gemini_api_key_configured": RuntimeConfig.has_api_key(),
        "preferred_model": RuntimeConfig.get_model(),
        "allowed_free_models": ALLOWED_FREE_GEMINI_MODELS,
    }


@app.post("/api/config/key")
def configure_gemini_key(req: ApiKeyConfigRequest):
    key = req.gemini_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
    if req.preferred_model:
        if req.preferred_model not in ALLOWED_FREE_GEMINI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Only free-tier Gemini models are permitted: {', '.join(ALLOWED_FREE_GEMINI_MODELS)}",
            )
        RuntimeConfig.set_model(req.preferred_model)
    RuntimeConfig.set_api_key(key)
    return {
        "status": "configured",
        "gemini_api_key_configured": True,
        "preferred_model": RuntimeConfig.get_model(),
    }


@app.post("/api/config/test-gemini")
def test_gemini_connection(preferred_model: Optional[str] = None):
    try:
        res = verify_gemini_connectivity(preferred_model=preferred_model)
        return res
    except GeminiServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _run_analysis_pipeline(
    job_id: str,
    source_type: str,
    temp_upload_path: Optional[Path] = None,
    original_filename: Optional[str] = None,
    youtube_url: Optional[str] = None,
    uploaded_subtitle_path: Optional[Path] = None,
    language_hint: str = "auto",
    gemini_model: Optional[str] = None,
    auto_render_top_clips: bool = False,
) -> None:
    """Background worker executing Media Extraction -> Transcription -> Gemini Best Moment Detection -> Optional Auto-Clipping."""
    try:
        update_job_progress(
            job_id,
            JobStatus.EXTRACTING_MEDIA,
            15,
            f"Extracting and validating video & audio streams ({source_type})...",
        )

        subtitle_path = uploaded_subtitle_path
        if source_type == "upload":
            if not temp_upload_path or not original_filename:
                raise VideoInputError("Missing uploaded file reference.")
            metadata = ingest_uploaded_file(temp_upload_path, original_filename, job_id)
        else:
            if not youtube_url:
                raise VideoInputError("Missing YouTube URL.")
            metadata, yt_sub = ingest_youtube_url(youtube_url, job_id)
            if not subtitle_path and yt_sub:
                subtitle_path = yt_sub

        job = get_job(job_id)
        if not job:
            return
        job.metadata = metadata
        save_job(job)

        # Step 2: Transcription & Word-Level Alignment
        update_job_progress(
            job_id,
            JobStatus.TRANSCRIBING,
            40,
            f"Transcribing audio ({metadata.duration:.1f}s) with word-level timestamps (Hindi/English/Hinglish)...",
        )
        wav_path = Path(metadata.audio_path) if metadata.audio_path else None
        if not wav_path or not wav_path.exists():
            raise VideoInputError("Extracted audio WAV file is missing.")

        transcript, model_used_t = transcribe_video_audio(
            wav_path=wav_path,
            duration=metadata.duration,
            language_hint=language_hint,
            subtitle_path=subtitle_path,
            preferred_gemini_model=gemini_model,
        )
        job = get_job(job_id)
        if not job:
            return
        job.transcript = transcript
        if model_used_t:
            job.gemini_model_used = model_used_t
        save_job(job)

        # Step 3: Gemini Best Moment Detection
        update_job_progress(
            job_id,
            JobStatus.ANALYZING_MOMENTS,
            70,
            "Analyzing transcript with Gemini Free-Tier model to identify highest-engagement moments...",
        )
        moments, model_used_m = detect_best_moments_with_gemini(
            transcript=transcript,
            video_duration=metadata.duration,
            video_title=metadata.original_name,
            preferred_model=gemini_model,
        )
        job = get_job(job_id)
        if not job:
            return
        job.moments = moments
        job.gemini_model_used = model_used_m or job.gemini_model_used
        save_job(job)

        if auto_render_top_clips and moments:
            update_job_progress(
                job_id,
                JobStatus.RENDERING_CLIPS,
                85,
                f"Rendering top {min(2, len(moments))} vertical 9:16 clips with subject framing & animated captions...",
            )
            rendered = []
            for m in moments[:2]:
                clip = render_vertical_clip(
                    job_id=job_id,
                    clip_id=m.clip_id,
                    title=m.title,
                    metadata=metadata,
                    segments=transcript.segments,
                    start_time=m.start_time,
                    end_time=m.end_time,
                    target_width=1080,
                    target_height=1920,
                    burn_captions=True,
                    subject_tracking=True,
                )
                rendered.append(clip)
            job = get_job(job_id)
            if job:
                job.clips = rendered
                save_job(job)
            update_job_progress(
                job_id,
                JobStatus.COMPLETED,
                100,
                f"Analysis and {len(rendered)} vertical 9:16 MP4 clip(s) completed.",
            )
        else:
            update_job_progress(
                job_id,
                JobStatus.READY_FOR_CLIPPING,
                100,
                f"Identified {len(moments)} engaging moment(s). Select clips to render 9:16 MP4s.",
            )

    except (VideoInputError, GeminiServiceError, ClipGenerationError) as exc:
        update_job_progress(
            job_id,
            JobStatus.FAILED,
            100,
            f"Pipeline error: {exc}",
            error_message=str(exc),
        )
    except Exception as exc:
        update_job_progress(
            job_id,
            JobStatus.FAILED,
            100,
            f"Unexpected processing error: {exc}",
            error_message=str(exc),
        )


@app.post("/api/jobs/upload")
async def create_upload_job(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    subtitle_file: Optional[UploadFile] = File(default=None),
    language_hint: str = Form(default="auto"),
    gemini_model: str = Form(default="gemini-2.5-flash"),
    auto_render_top_clips: bool = Form(default=False),
):
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded video file must have a filename.")
    ext = Path(video_file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported extensions: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
        )
    if gemini_model and gemini_model not in ALLOWED_FREE_GEMINI_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Only free-tier Gemini models are allowed: {', '.join(ALLOWED_FREE_GEMINI_MODELS)}",
        )

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    temp_video_path = TEMP_DIR / f"{job_id}_raw{ext}"
    with temp_video_path.open("wb") as f_out:
        while chunk := await video_file.read(1024 * 1024):
            f_out.write(chunk)

    temp_sub_path: Optional[Path] = None
    if subtitle_file and subtitle_file.filename:
        sub_ext = Path(subtitle_file.filename).suffix.lower()
        if sub_ext in {".srt", ".vtt"}:
            temp_sub_path = TEMP_DIR / f"{job_id}_subs{sub_ext}"
            with temp_sub_path.open("wb") as s_out:
                s_out.write(await subtitle_file.read())

    job = create_job(job_id, step_message=f"Uploaded '{video_file.filename}', queued for analysis")
    background_tasks.add_task(
        _run_analysis_pipeline,
        job_id=job_id,
        source_type="upload",
        temp_upload_path=temp_video_path,
        original_filename=video_file.filename,
        uploaded_subtitle_path=temp_sub_path,
        language_hint=language_hint,
        gemini_model=gemini_model,
        auto_render_top_clips=auto_render_top_clips,
    )
    return job


@app.post("/api/jobs/youtube")
def create_youtube_job(req: YouTubeIngestRequest, background_tasks: BackgroundTasks):
    if req.gemini_model and req.gemini_model not in ALLOWED_FREE_GEMINI_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Only free-tier Gemini models are allowed: {', '.join(ALLOWED_FREE_GEMINI_MODELS)}",
        )
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = create_job(job_id, step_message=f"Queued YouTube URL: {req.url}")
    background_tasks.add_task(
        _run_analysis_pipeline,
        job_id=job_id,
        source_type="youtube",
        youtube_url=req.url,
        language_hint=req.language_hint or "auto",
        gemini_model=req.gemini_model or RuntimeConfig.get_model(),
        auto_render_top_clips=req.auto_render_top_clips,
    )
    return job


@app.get("/api/jobs")
def get_all_jobs():
    return {"jobs": list_jobs(limit=30)}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _run_clip_render_worker(job_id: str, req: RenderClipsRequest) -> None:
    job = get_job(job_id)
    if not job or not job.metadata or not job.transcript:
        update_job_progress(
            job_id,
            JobStatus.FAILED,
            100,
            "Cannot render clips: job metadata or transcript is missing.",
            error_message="Missing metadata or transcript.",
        )
        return

    w_str, h_str = req.target_resolution.split("x")
    target_w, target_h = int(w_str), int(h_str)

    try:
        total = len(req.clips)
        existing_map = {c.clip_id: c for c in job.clips}

        for idx, item in enumerate(req.clips, start=1):
            pct = 15 + int((idx - 1) / max(1, total) * 80)
            update_job_progress(
                job_id,
                JobStatus.RENDERING_CLIPS,
                pct,
                f"Rendering clip {idx}/{total} ('{item.title}', {item.start_time:.1f}s - {item.end_time:.1f}s) in 9:16 ({target_w}x{target_h})...",
            )
            rendered_clip = render_vertical_clip(
                job_id=job_id,
                clip_id=item.clip_id,
                title=item.title,
                metadata=job.metadata,
                segments=job.transcript.segments,
                start_time=item.start_time,
                end_time=item.end_time,
                target_width=target_w,
                target_height=target_h,
                burn_captions=item.burn_captions,
                subject_tracking=item.subject_tracking,
            )
            existing_map[item.clip_id] = rendered_clip
            job = get_job(job_id)
            if job:
                job.clips = list(existing_map.values())
                save_job(job)

        update_job_progress(
            job_id,
            JobStatus.COMPLETED,
            100,
            f"Successfully rendered and verified {len(req.clips)} playable 9:16 MP4 clip(s).",
        )
    except Exception as exc:
        update_job_progress(
            job_id,
            JobStatus.FAILED,
            100,
            f"Clip rendering failed: {exc}",
            error_message=str(exc),
        )


@app.post("/api/jobs/{job_id}/render")
def render_selected_clips(
    job_id: str, req: RenderClipsRequest, background_tasks: BackgroundTasks
):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if not job.metadata or not job.transcript:
        raise HTTPException(
            status_code=400,
            detail="Job has not completed video extraction and transcription yet.",
        )
    if not req.clips:
        raise HTTPException(status_code=400, detail="At least one clip must be selected for rendering.")

    # Validate timestamps against source video duration before starting
    for c in req.clips:
        if c.start_time < 0 or c.end_time <= c.start_time:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clip timestamps ({c.start_time}s - {c.end_time}s) for '{c.title}'.",
            )
        if c.start_time >= job.metadata.duration:
            raise HTTPException(
                status_code=400,
                detail=f"Clip start time ({c.start_time}s) exceeds video duration ({job.metadata.duration}s).",
            )

    update_job_progress(
        job_id,
        JobStatus.RENDERING_CLIPS,
        10,
        f"Queued {len(req.clips)} clip(s) for FFmpeg 9:16 rendering and caption burn-in...",
    )
    background_tasks.add_task(_run_clip_render_worker, job_id=job_id, req=req)
    return get_job(job_id)


def _resolve_clip_file(job_id: str, clip_id: str) -> Path:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    clip_meta = next((c for c in job.clips if c.clip_id == clip_id), None)
    if not clip_meta:
        raise HTTPException(status_code=404, detail=f"Clip '{clip_id}' not found in job '{job_id}'.")
    file_path = CLIPS_DIR / job_id / clip_meta.file_name
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise HTTPException(status_code=404, detail=f"Exported MP4 file missing on disk: {clip_meta.file_name}")
    return file_path


@app.get("/api/jobs/{job_id}/clips/{clip_id}/download")
def download_clip_mp4(job_id: str, clip_id: str):
    file_path = _resolve_clip_file(job_id, clip_id)
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
        headers={"Content-Disposition": f'attachment; filename="{file_path.name}"'},
    )


@app.get("/api/jobs/{job_id}/clips/{clip_id}/stream")
def stream_clip_mp4(job_id: str, clip_id: str):
    file_path = _resolve_clip_file(job_id, clip_id)
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
    )


# Serve production frontend build if available
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def serve_spa_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return HTMLResponse("<h1>ArgonClip API is running. Frontend dist/index.html not found.</h1>")
