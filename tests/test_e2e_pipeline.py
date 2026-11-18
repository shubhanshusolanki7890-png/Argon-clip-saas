import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.main import app
from backend.models import BestMoment, JobStatus
from backend.services.job_store import get_job, save_job
from backend.services.video_input import probe_media_file


def create_synthetic_test_video_and_subs(out_dir: Path):
    """
    Create a genuine 16:9 landscape MP4 video (1280x720, 12 seconds, 30 fps) with:
    - A moving subject/speaker indicator that pans from left (x=340) to right (x=940) and back
    - Genuine modulated speech-envelope audio bursts matching word rhythms
    - Multilingual Hinglish + Hindi + English SRT subtitle track
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_video = out_dir / "raw_visual.mp4"
    raw_audio = out_dir / "speech_envelope.wav"
    final_video = out_dir / "sample_long_video.mp4"
    srt_file = out_dir / "sample_long_video.srt"

    width, height, fps, duration = 1280, 720, 30, 12
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video), fourcc, float(fps), (width, height))

    for f_idx in range(total_frames):
        t = f_idx / float(fps)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (32, 24, 18)

        # Subject moves horizontally across the 16:9 frame over time
        subject_cx = int(640 + 300 * np.sin(2 * np.pi * t / 12.0))
        subject_cy = 340

        cv2.circle(frame, (subject_cx, subject_cy - 70), 65, (210, 190, 175), -1)
        cv2.circle(frame, (subject_cx - 22, subject_cy - 80), 8, (25, 25, 25), -1)
        cv2.circle(frame, (subject_cx + 22, subject_cy - 80), 8, (25, 25, 25), -1)
        mouth_open = 6 + int(8 * abs(np.sin(t * 9.0)))
        cv2.ellipse(
            frame,
            (subject_cx, subject_cy - 45),
            (18, mouth_open),
            0,
            0,
            360,
            (40, 40, 180),
            -1,
        )
        cv2.rectangle(
            frame,
            (subject_cx - 110, subject_cy + 5),
            (subject_cx + 110, subject_cy + 260),
            (70, 130, 200),
            -1,
        )
        cv2.putText(
            frame,
            f"SRC 16:9 | t={t:.1f}s | subj_x={subject_cx}",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 220, 255),
            2,
        )
        writer.write(frame)

    writer.release()

    # Generate 16kHz WAV with modulated speech-like syllable bursts
    sr = 16000
    t_samples = np.linspace(0, duration, sr * duration, endpoint=False)
    envelope = (0.2 + 0.8 * (np.sin(2 * np.pi * 3.5 * t_samples) > -0.2)).astype(np.float32)
    carrier = (
        0.5 * np.sin(2 * np.pi * 190 * t_samples)
        + 0.3 * np.sin(2 * np.pi * 380 * t_samples)
        + 0.2 * np.sin(2 * np.pi * 570 * t_samples)
    )
    pcm = np.clip(carrier * envelope * 0.45 * 32767, -32767, 32767).astype(np.int16)
    with wave.open(str(raw_audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_video),
        "-i",
        str(raw_audio),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(final_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    srt_content = """1
00:00:00,200 --> 00:00:04,000
Namaste dosto aaj hum AI video clipping pipeline test kar rahe hain.

2
00:00:04,100 --> 00:00:08,200
This vertical clip automatically tracks the speaker and highlights each active word in yellow.

3
00:00:08,300 --> 00:00:11,800
असली MP4 क्लिप अब डाउनलोड के लिए पूरी तरह तैयार है।
"""
    srt_file.write_text(srt_content, encoding="utf-8")
    return final_video, srt_file


def run_all_tests():
    print("=== [Test 1] Frontend Production Build Check ===")
    build_proc = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(ROOT_DIR / "frontend"),
        capture_output=True,
        text=True,
        check=True,
    )
    print(build_proc.stdout.strip())
    assert (ROOT_DIR / "frontend" / "dist" / "index.html").exists()
    assert (ROOT_DIR / "frontend" / "dist" / "assets" / "app.js").exists()
    assert (ROOT_DIR / "frontend" / "dist" / "assets" / "styles.css").exists()

    client = TestClient(app)

    print("\n=== [Test 2] Backend Health, CORS & Free-Model Guardrail Check ===")
    r_health = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r_health.status_code == 200
    h_data = r_health.json()
    print("Health Endpoint:", json.dumps(h_data))
    assert h_data["ffmpeg_available"] is True
    assert h_data["ffprobe_available"] is True
    assert "access-control-allow-origin" in r_health.headers

    # Ensure paid/non-free Gemini model is rejected
    r_bad_model = client.post(
        "/api/config/key",
        json={"gemini_api_key": "test_dummy_key", "preferred_model": "gemini-1.5-pro"},
    )
    assert r_bad_model.status_code == 400
    print("Non-free model rejection verified:", r_bad_model.json()["detail"])

    # Ensure invalid YouTube URL is rejected cleanly
    r_bad_yt = client.post(
        "/api/jobs/youtube",
        json={"url": "https://example.com/not-a-youtube-video"},
    )
    assert r_bad_yt.status_code == 200
    bad_job_id = r_bad_yt.json()["job_id"]
    bad_job = client.get(f"/api/jobs/{bad_job_id}").json()
    assert bad_job["status"] == "failed"
    assert "Unsupported or invalid YouTube URL" in bad_job["error_message"]
    print("Unsupported URL error handling verified:", bad_job["error_message"])

    print("\n=== [Test 3] Generate Source 16:9 Video + Multilingual Speech Track ===")
    sample_dir = ROOT_DIR / "storage" / "temp" / "e2e_sample"
    src_mp4, src_srt = create_synthetic_test_video_and_subs(sample_dir)
    src_probe = probe_media_file(src_mp4)
    print("Generated 16:9 Source Video:", json.dumps(src_probe))
    assert src_probe["width"] == 1280 and src_probe["height"] == 720
    assert src_probe["has_audio"] is True

    print("\n=== [Test 4] Upload Video + Subtitle Track & Verify Multilingual Word Alignment ===")
    with src_mp4.open("rb") as vf, src_srt.open("rb") as sf:
        r_up = client.post(
            "/api/jobs/upload",
            files={
                "video_file": ("sample_long_video.mp4", vf, "video/mp4"),
                "subtitle_file": ("sample_long_video.srt", sf, "text/plain"),
            },
            data={
                "language_hint": "hinglish",
                "gemini_model": "gemini-2.5-flash",
                "auto_render_top_clips": "false",
            },
        )
    assert r_up.status_code == 200
    job_id = r_up.json()["job_id"]
    job_obj = get_job(job_id)
    assert job_obj is not None
    assert job_obj.metadata is not None
    assert job_obj.transcript is not None
    assert len(job_obj.transcript.segments) == 3
    total_words = sum(len(s.words) for s in job_obj.transcript.segments)
    print(
        f"Transcription verified: language={job_obj.transcript.language}, "
        f"segments={len(job_obj.transcript.segments)}, word_timestamps={total_words}"
    )
    assert total_words >= 20

    # Populate verified moments on the job record and test the full FFmpeg 9:16 render + ASS caption burn pipeline
    job_obj.moments = [
        BestMoment(
            clip_id="clip_hinglish_1",
            title="Hinglish & English AI Clipping Showcase",
            start_time=0.0,
            end_time=8.2,
            duration=8.2,
            score=95,
            hook_strength=94,
            storytelling_score=92,
            standalone_value=96,
            language="hinglish",
            rationale="Strong Hinglish opening hook followed by clear explanation of subject tracking and yellow word highlights.",
            transcript_excerpt=job_obj.transcript.full_text,
            selected=True,
        ),
        BestMoment(
            clip_id="clip_hindi_2",
            title="Hindi Devanagari Caption Burn Test",
            start_time=4.1,
            end_time=11.8,
            duration=7.7,
            score=91,
            hook_strength=89,
            storytelling_score=90,
            standalone_value=93,
            language="hi",
            rationale="Validates mixed English and Hindi Devanagari word-by-word yellow caption highlighting.",
            transcript_excerpt=job_obj.transcript.segments[2].text,
            selected=True,
        ),
    ]
    job_obj.status = JobStatus.READY_FOR_CLIPPING
    job_obj.error_message = None
    save_job(job_obj)

    print("\n=== [Test 5] Render 9:16 Vertical Clips with Subject Framing & White/Yellow Animated Captions ===")
    r_render = client.post(
        f"/api/jobs/{job_id}/render",
        json={
            "target_resolution": "1080x1920",
            "clips": [
                {
                    "clip_id": "clip_hinglish_1",
                    "title": "Hinglish & English AI Clipping Showcase",
                    "start_time": 0.0,
                    "end_time": 8.2,
                    "burn_captions": True,
                    "subject_tracking": True,
                },
                {
                    "clip_id": "clip_hindi_2",
                    "title": "Hindi Devanagari Caption Burn Test",
                    "start_time": 4.1,
                    "end_time": 11.8,
                    "burn_captions": True,
                    "subject_tracking": True,
                },
            ],
        },
    )
    assert r_render.status_code == 200
    # Fetch post-background-task job state
    final_job = client.get(f"/api/jobs/{job_id}").json()
    assert final_job["status"] == "completed", f"Render failed: {final_job}"
    assert len(final_job["clips"]) == 2

    for clip_meta in final_job["clips"]:
        print("Verified Rendered Clip:", json.dumps(clip_meta))
        assert clip_meta["width"] == 1080
        assert clip_meta["height"] == 1920
        assert clip_meta["aspect_ratio"] == "9:16"
        assert clip_meta["verified_playable"] is True
        assert clip_meta["caption_word_count"] > 0

        # Test HTTP Download endpoint
        dl_res = client.get(clip_meta["download_url"])
        assert dl_res.status_code == 200
        assert dl_res.headers["content-type"] == "video/mp4"
        assert len(dl_res.content) > 50_000

    # Inspect rendered frames in bottom caption zone to verify white & yellow burned-in caption pixels exist
    exported_mp4 = ROOT_DIR / "storage" / "clips" / job_id / final_job["clips"][0]["file_name"]
    cap = cv2.VideoCapture(str(exported_mp4))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1500.0)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None
    assert frame.shape == (1920, 1080, 3)

    # Bottom caption region (y: 1450..1800)
    caption_zone = frame[1450:1800, 100:980]
    b, g, r = caption_zone[:, :, 0], caption_zone[:, :, 1], caption_zone[:, :, 2]
    white_pixels = int(np.sum((r > 210) & (g > 210) & (b > 210)))
    yellow_pixels = int(np.sum((r > 200) & (g > 200) & (b < 90)))
    print(
        f"Caption Pixel Verification at t=1.5s -> White pixels: {white_pixels}, Yellow highlight pixels: {yellow_pixels}"
    )
    assert white_pixels > 100, "Expected white normal caption pixels in bottom region"
    assert yellow_pixels > 50, "Expected yellow highlighted active word pixels in bottom region"

    print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY!")
    return exported_mp4


if __name__ == "__main__":
    run_all_tests()
