import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from backend.config import (
    MAX_UPLOAD_SIZE_BYTES,
    SUPPORTED_VIDEO_EXTENSIONS,
    TEMP_DIR,
    UPLOADS_DIR,
)
from backend.models import VideoMetadata

YOUTUBE_URL_PATTERN = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|live/|embed/)|youtu\.be/)[\w\-]{6,}",
    re.IGNORECASE,
)


class VideoInputError(Exception):
    """Raised when video input validation, download, or stream probing fails."""


def is_valid_youtube_url(url: str) -> bool:
    url = url.strip()
    if not YOUTUBE_URL_PATTERN.match(url):
        return False
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower() in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }


def probe_media_file(file_path: Path) -> dict:
    """Run ffprobe on a media file and return parsed stream and format metadata."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise VideoInputError(f"Media file does not exist or is empty: {file_path.name}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "Unknown ffprobe error"
        raise VideoInputError(f"Failed to probe media file '{file_path.name}': {stderr}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoInputError(f"Invalid ffprobe output for '{file_path.name}': {exc}") from exc

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        raise VideoInputError(f"File '{file_path.name}' does not contain a valid video stream.")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise VideoInputError(f"Invalid video dimensions ({width}x{height}) in '{file_path.name}'.")

    fps_raw = video_stream.get("r_frame_rate", "30/1")
    try:
        if "/" in fps_raw:
            num, den = fps_raw.split("/", 1)
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_raw)
    except Exception:
        fps = 30.0

    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0.0) or 0.0)
    if duration <= 0.5:
        raise VideoInputError(
            f"Video duration ({duration:.2f}s) is too short. Minimum supported duration is 1.0 second."
        )

    return {
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "has_audio": audio_stream is not None,
        "file_size_bytes": int(fmt.get("size", file_path.stat().st_size)),
    }


def extract_audio_track(video_path: Path, job_id: str, has_audio: bool = True) -> Path:
    """
    Extract 16kHz mono WAV audio from the video file for transcription and Gemini analysis.
    If the source video has no audio track, generates a silent 16kHz WAV matching the duration
    so downstream audio pipelines do not crash.
    """
    output_wav = TEMP_DIR / f"{job_id}_audio.wav"
    if has_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0:
            return output_wav

    # Fallback if no audio stream exists
    probe = probe_media_file(video_path)
    dur = max(1.0, probe["duration"])
    cmd_silent = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-t",
        f"{dur:.3f}",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    proc_s = subprocess.run(cmd_silent, capture_output=True, text=True, timeout=60)
    if proc_s.returncode != 0 or not output_wav.exists():
        raise VideoInputError(f"Audio extraction failed: {proc_s.stderr.strip()}")
    return output_wav


def ingest_uploaded_file(
    temp_upload_path: Path, original_filename: str, job_id: str
) -> VideoMetadata:
    """Validate an uploaded video file, move it to job storage, probe streams, and extract audio."""
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoInputError(
            f"Unsupported file extension '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    size = temp_upload_path.stat().st_size
    if size == 0:
        raise VideoInputError("Uploaded video file is empty (0 bytes).")
    if size > MAX_UPLOAD_SIZE_BYTES:
        raise VideoInputError(
            f"Uploaded file size ({size / (1024 * 1024):.1f} MB) exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    dest_path = UPLOADS_DIR / f"{job_id}{ext}"
    shutil.move(str(temp_upload_path), str(dest_path))

    try:
        info = probe_media_file(dest_path)
        audio_path = extract_audio_track(dest_path, job_id, has_audio=info["has_audio"])
    except Exception:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        raise

    return VideoMetadata(
        source_type="upload",
        original_name=original_filename,
        source_url=None,
        video_path=str(dest_path),
        audio_path=str(audio_path),
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        video_codec=info["video_codec"],
        audio_codec=info["audio_codec"],
        has_audio=info["has_audio"],
        file_size_bytes=info["file_size_bytes"],
    )


def ingest_youtube_url(url: str, job_id: str) -> Tuple[VideoMetadata, Optional[Path]]:
    """
    Validate and download a YouTube URL using yt-dlp (Python module or CLI binary).
    Also attempts to fetch automatic/manual subtitles (.vtt/.json3) if available.
    Raises VideoInputError with actionable diagnostic details on failure.
    """
    if not is_valid_youtube_url(url):
        raise VideoInputError(
            f"Unsupported or invalid YouTube URL: '{url}'. Please provide a valid youtube.com/watch?v=..., youtube.com/shorts/..., or youtu.be/... link, or upload an MP4/MOV/MKV/WEBM file directly."
        )

    output_template = str(UPLOADS_DIR / f"{job_id}.%(ext)s")
    final_video_path: Optional[Path] = None
    video_title = f"youtube_{job_id}"

    # Method 1: Try yt_dlp Python library if installed
    try:
        import yt_dlp  # type: ignore

        ydl_opts = {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "hi.*", "en", "hi"],
            "socket_timeout": 25,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get("title") or video_title
            downloaded = ydl.prepare_filename(info_dict)
            candidate = Path(downloaded)
            if not candidate.exists():
                candidate = UPLOADS_DIR / f"{job_id}.mp4"
            if candidate.exists():
                final_video_path = candidate
    except ImportError:
        # Method 2: Try yt-dlp CLI binary if installed
        yt_dlp_bin = shutil.which("yt-dlp")
        if not yt_dlp_bin:
            raise VideoInputError(
                "yt-dlp is not installed in the current runtime environment (`pip install yt-dlp`). "
                "To process videos in this environment without yt-dlp, please upload the video file (.mp4, .mov, .mkv, .webm) directly."
            )
        cmd = [
            yt_dlp_bin,
            "--no-playlist",
            "-f",
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            "en.*,hi.*",
            "-o",
            output_template,
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown yt-dlp error"
            raise VideoInputError(
                f"YouTube extraction failed via yt-dlp: {err_msg}. "
                "If YouTube is blocking automated requests or outbound network is restricted, upload the MP4 file directly."
            )
    except Exception as exc:
        raise VideoInputError(
            f"YouTube extraction failed: {exc}. "
            "If YouTube blocked the request or outbound network access is restricted, please upload the video file directly."
        ) from exc

    if not final_video_path:
        matches = [
            p
            for p in UPLOADS_DIR.glob(f"{job_id}.*")
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        if matches:
            final_video_path = matches[0]

    if not final_video_path or not final_video_path.exists():
        raise VideoInputError("YouTube video download completed without producing a valid media file.")

    # Check if any subtitle file was downloaded alongside the video
    sub_matches = list(UPLOADS_DIR.glob(f"{job_id}*.vtt")) + list(
        UPLOADS_DIR.glob(f"{job_id}*.srt")
    )
    subtitle_path = sub_matches[0] if sub_matches else None

    info = probe_media_file(final_video_path)
    audio_path = extract_audio_track(final_video_path, job_id, has_audio=info["has_audio"])

    metadata = VideoMetadata(
        source_type="youtube",
        original_name=video_title,
        source_url=url,
        video_path=str(final_video_path),
        audio_path=str(audio_path),
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        video_codec=info["video_codec"],
        audio_codec=info["audio_codec"],
        has_audio=info["has_audio"],
        file_size_bytes=info["file_size_bytes"],
    )
    return metadata, subtitle_path
