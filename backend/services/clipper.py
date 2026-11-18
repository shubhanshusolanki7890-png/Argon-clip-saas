import re
import subprocess
from pathlib import Path
from typing import List

from backend.config import CLIPS_DIR, TEMP_DIR
from backend.models import GeneratedClip, TranscriptSegment, VideoMetadata
from backend.services.captions import generate_animated_ass_captions
from backend.services.framing import (
    analyze_subject_trajectory,
    build_ffmpeg_framing_filter,
)
from backend.services.video_input import probe_media_file


class ClipGenerationError(Exception):
    """Raised when FFmpeg clip rendering or output verification fails."""


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for safe embedding in an FFmpeg `ass=...` filter string."""
    s = str(path.resolve())
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")
    return s


def render_vertical_clip(
    job_id: str,
    clip_id: str,
    title: str,
    metadata: VideoMetadata,
    segments: List[TranscriptSegment],
    start_time: float,
    end_time: float,
    target_width: int = 1080,
    target_height: int = 1920,
    burn_captions: bool = True,
    subject_tracking: bool = True,
) -> GeneratedClip:
    """
    Render a genuine 9:16 vertical MP4 clip with:
    1. Timestamp validation against actual source duration.
    2. Intelligent OpenCV subject-aware framing (face/motion tracking crop + Lanczos scaling).
    3. Burned-in animated ASS captions (white normal words, yellow active word near bottom).
    4. Audio-video sync preservation and post-render ffprobe verification.
    """
    src_path = Path(metadata.video_path)
    if not src_path.exists():
        raise ClipGenerationError(f"Source video file not found: {src_path}")

    # Clamp and validate timestamps against actual source video duration
    valid_start = max(0.0, min(float(start_time), metadata.duration - 0.5))
    valid_end = max(valid_start + 0.5, min(float(end_time), metadata.duration))
    clip_duration = round(valid_end - valid_start, 3)

    job_clip_dir = CLIPS_DIR / job_id
    job_clip_dir.mkdir(parents=True, exist_ok=True)

    safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", clip_id).strip("_") or "clip"
    out_filename = f"{job_id}_{safe_slug}.mp4"
    out_mp4_path = job_clip_dir / out_filename

    # 1. Subject-Aware Framing Analysis
    if subject_tracking:
        keyframes, framing_info = analyze_subject_trajectory(
            video_path=src_path,
            start_time=valid_start,
            end_time=valid_end,
            sample_interval_sec=0.5,
        )
    else:
        from backend.models import FramingKeyframe

        keyframes = [
            FramingKeyframe(timestamp=0.0, center_x_norm=0.5, detection_method="center_locked")
        ]
        framing_info = {"mode": "static_center", "face_hit_ratio": 0.0, "mean_center_x": 0.5}

    vf_chain = build_ffmpeg_framing_filter(
        src_width=metadata.width,
        src_height=metadata.height,
        target_width=target_width,
        target_height=target_height,
        keyframes=keyframes,
        framing_info=framing_info,
    )

    # 2. Animated Word-Level ASS Captions
    caption_word_count = 0
    ass_path = TEMP_DIR / f"{job_id}_{safe_slug}.ass"
    if burn_captions:
        _, caption_word_count = generate_animated_ass_captions(
            segments=segments,
            clip_start=valid_start,
            clip_end=valid_end,
            output_ass_path=ass_path,
            target_width=target_width,
            target_height=target_height,
        )
        if ass_path.exists() and caption_word_count > 0:
            escaped_ass = _escape_ffmpeg_filter_path(ass_path)
            vf_chain = f"{vf_chain},ass='{escaped_ass}'"

    # 3. Execute FFmpeg Render
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{valid_start:.3f}",
        "-t",
        f"{clip_duration:.3f}",
        "-i",
        str(src_path),
        "-vf",
        vf_chain,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
    ]

    if metadata.has_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
    else:
        # Attach silent AAC track if source had no audio so all exported MP4s have valid audio/video streams
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{valid_start:.3f}",
            "-t",
            f"{clip_duration:.3f}",
            "-i",
            str(src_path),
            "-f",
            "lavfi",
            "-t",
            f"{clip_duration:.3f}",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf",
            vf_chain,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
        ]

    cmd += ["-movflags", "+faststart", str(out_mp4_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise ClipGenerationError(
            f"FFmpeg clip rendering failed for '{clip_id}': {proc.stderr.strip()}"
        )

    # 4. Verify Exported MP4 File Integrity & Playability via ffprobe
    verified_info = probe_media_file(out_mp4_path)
    if verified_info["width"] != target_width or verified_info["height"] != target_height:
        raise ClipGenerationError(
            f"Exported clip dimensions ({verified_info['width']}x{verified_info['height']}) "
            f"do not match target 9:16 resolution ({target_width}x{target_height})."
        )

    framing_summary = (
        f"{framing_info.get('mode')} (center_x={framing_info.get('mean_center_x', 0.5):.2f}, "
        f"face_detect_rate={int(framing_info.get('face_hit_ratio', 0.0) * 100)}%)"
    )

    return GeneratedClip(
        clip_id=clip_id,
        title=title,
        start_time=round(valid_start, 2),
        end_time=round(valid_end, 2),
        duration=verified_info["duration"],
        width=verified_info["width"],
        height=verified_info["height"],
        aspect_ratio="9:16",
        file_name=out_filename,
        file_size_bytes=verified_info["file_size_bytes"],
        download_url=f"/api/jobs/{job_id}/clips/{clip_id}/download",
        stream_url=f"/api/jobs/{job_id}/clips/{clip_id}/stream",
        framing_mode=str(framing_info.get("mode", "locked_subject_crop")),
        framing_summary=framing_summary,
        caption_word_count=caption_word_count,
        has_audio=verified_info["has_audio"],
        verified_playable=True,
    )
