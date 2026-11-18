import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.config import ALLOWED_FREE_GEMINI_MODELS, RuntimeConfig, TEMP_DIR
from backend.models import BestMoment, TranscriptResult, TranscriptSegment, WordTimestamp

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta/files"


class GeminiServiceError(Exception):
    """Raised when Gemini API calls fail, rate-limit out, or lack valid credentials."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _get_fallback_model_chain(preferred_model: Optional[str] = None) -> List[str]:
    start_model = preferred_model or RuntimeConfig.get_model()
    if start_model not in ALLOWED_FREE_GEMINI_MODELS:
        start_model = "gemini-2.5-flash"
    chain = [start_model] + [m for m in ALLOWED_FREE_GEMINI_MODELS if m != start_model]
    return chain


def _extract_json_payload(raw_text: str) -> Any:
    """Safely parse JSON from Gemini response text, stripping markdown code fences if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if match:
            return json.loads(match.group(1))
        raise


def _compress_audio_for_gemini(wav_path: Path, start_sec: float = 0.0, duration_sec: Optional[float] = None) -> Path:
    """Convert WAV segment to compact 48kbps mono MP3 for fast multimodal Gemini transmission."""
    suffix = f"_{int(start_sec)}_{int(duration_sec or 0)}.mp3"
    mp3_path = TEMP_DIR / f"{wav_path.stem}{suffix}"
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}"]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    cmd += [
        "-i",
        str(wav_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not mp3_path.exists():
        raise GeminiServiceError(f"Failed to encode audio chunk for Gemini: {proc.stderr.strip()}")
    return mp3_path


def call_gemini_generate_content(
    contents: List[Dict[str, Any]],
    system_instruction: Optional[str] = None,
    response_mime_type: str = "application/json",
    preferred_model: Optional[str] = None,
    temperature: float = 0.2,
    timeout_sec: int = 120,
) -> Tuple[Any, str]:
    """
    Execute a generateContent request against the free-tier Gemini models with automatic
    rate-limit backoff (HTTP 429 / 503) and model fallback across ALLOWED_FREE_GEMINI_MODELS.
    Returns (parsed_response, model_used).
    """
    api_key = RuntimeConfig.get_api_key()
    if not api_key:
        raise GeminiServiceError(
            "GEMINI_API_KEY is not configured. Set GEMINI_API_KEY in the environment or via POST /api/config/key.",
            status_code=401,
        )

    models_to_try = _get_fallback_model_chain(preferred_model)
    last_error = "Unknown Gemini API error"
    last_status = 500

    for model_name in models_to_try:
        url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": response_mime_type,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=timeout_sec)
            except requests.RequestException as exc:
                last_error = f"Network error connecting to Gemini API ({model_name}): {exc}"
                last_status = 503
                break

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    last_error = f"Gemini ({model_name}) returned no candidates: {data}"
                    break
                parts = candidates[0].get("content", {}).get("parts", [])
                text_out = "".join(p.get("text", "") for p in parts).strip()
                if not text_out:
                    last_error = f"Gemini ({model_name}) returned empty text output."
                    break
                if response_mime_type == "application/json":
                    return _extract_json_payload(text_out), model_name
                return text_out, model_name

            last_status = resp.status_code
            try:
                err_body = resp.json().get("error", {})
                err_msg = err_body.get("message", resp.text)
            except Exception:
                err_msg = resp.text

            if resp.status_code in (400, 401, 403) and "API_KEY_INVALID" in err_msg.upper():
                raise GeminiServiceError(f"Invalid Gemini API key: {err_msg}", status_code=401)

            if resp.status_code == 429 or "RESOURCE_EXHAUSTED" in err_msg.upper():
                last_error = f"Quota/rate limit reached on free model {model_name}: {err_msg}"
                if attempt == 0:
                    time.sleep(2.0)
                    continue
                # Move to next free-tier model in fallback chain
                break

            if resp.status_code >= 500:
                last_error = f"Gemini server error ({resp.status_code}) on {model_name}: {err_msg}"
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break

            last_error = f"Gemini API error ({resp.status_code}) on {model_name}: {err_msg}"
            break

    raise GeminiServiceError(last_error, status_code=last_status)


def verify_gemini_connectivity(preferred_model: Optional[str] = None) -> Dict[str, Any]:
    """Perform a live connectivity check against the configured free-tier Gemini model."""
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": 'Return a JSON object with keys: {"status": "ok", "ready": true}'
                }
            ],
        }
    ]
    result, model_used = call_gemini_generate_content(
        contents=contents,
        response_mime_type="application/json",
        preferred_model=preferred_model,
        temperature=0.0,
        timeout_sec=20,
    )
    return {"connected": True, "model_used": model_used, "response": result}


def transcribe_audio_chunk_with_gemini(
    audio_wav_path: Path,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    language_hint: str = "auto",
    preferred_model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Transcribe a single audio chunk using Gemini multimodal audio input.
    Preserves Hindi, English, and mixed Hinglish speech verbatim.
    """
    mp3_chunk = _compress_audio_for_gemini(
        audio_wav_path, start_sec=chunk_start_sec, duration_sec=chunk_duration_sec
    )
    try:
        audio_bytes = mp3_chunk.read_bytes()
        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
    finally:
        mp3_chunk.unlink(missing_ok=True)

    lang_instruction = (
        "Preserve the exact spoken language verbatim (whether English, Hindi in Devanagari or Roman script, or mixed Hinglish). "
        "Do NOT translate Hindi/Hinglish into pure English or vice versa."
    )
    if language_hint and language_hint != "auto":
        lang_instruction += f" Language hint provided by user: {language_hint}."

    prompt = f"""You are an accurate speech-to-text transcription engine with timestamp alignment.
Listen to this audio clip (duration: {chunk_duration_sec:.2f} seconds) and produce a complete, verbatim transcript broken into phrase segments of 2 to 6 seconds each.
{lang_instruction}

Return strictly valid JSON with this exact schema:
{{
  "language": "en | hi | hinglish",
  "segments": [
    {{
      "start": 0.0,
      "end": 3.4,
      "text": "exact spoken words in this segment",
      "language": "en | hi | hinglish",
      "words": [
        {{"word": "exact", "start": 0.0, "end": 0.45}},
        {{"word": "spoken", "start": 0.45, "end": 0.95}}
      ]
    }}
  ]
}}

Rules:
1. All timestamps ("start" and "end") must be in seconds relative to the start of this audio chunk (between 0.0 and {chunk_duration_sec:.2f}).
2. Every segment MUST include a "words" array with start and end timestamps for each individual word in "text".
3. Never invent speech that is not present in the audio. If there is silence, skip the silent interval.
"""

    contents = [
        {
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "audio/mp3", "data": b64_audio}},
                {"text": prompt},
            ],
        }
    ]
    parsed, model_used = call_gemini_generate_content(
        contents=contents,
        response_mime_type="application/json",
        preferred_model=preferred_model,
        temperature=0.1,
        timeout_sec=180,
    )
    detected_lang = parsed.get("language", "en") if isinstance(parsed, dict) else "en"
    raw_segments = parsed.get("segments", []) if isinstance(parsed, dict) else []
    return raw_segments, detected_lang, model_used


def detect_best_moments_with_gemini(
    transcript: TranscriptResult,
    video_duration: float,
    video_title: str = "Video",
    preferred_model: Optional[str] = None,
) -> Tuple[List[BestMoment], str]:
    """
    Use Gemini Free-Tier model to evaluate the timestamped transcript (Hindi, English, or Hinglish)
    and identify the strongest self-contained short-form vertical clip moments.
    Strictly validates and clamps all timestamps against [0, video_duration].
    """
    if not transcript.segments:
        raise GeminiServiceError(
            "Cannot detect best moments: transcript contains no spoken segments.",
            status_code=422,
        )

    formatted_lines = []
    for seg in transcript.segments:
        formatted_lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] ({seg.language}) {seg.text}")
    transcript_block = "\n".join(formatted_lines)

    min_clip_dur = min(5.0, max(2.0, video_duration * 0.25))
    max_clip_dur = min(60.0, video_duration)

    system_prompt = (
        "You are an expert short-form video editor and viral content strategist fluent in English, Hindi, and Hinglish. "
        "Your job is to identify the most engaging, self-contained moments from a timestamped video transcript for vertical 9:16 Reels/Shorts."
    )

    user_prompt = f"""Analyze the following timestamped transcript for "{video_title}" (Total Video Duration: {video_duration:.2f} seconds; Primary Language: {transcript.language}).

Identify between 1 and 5 of the best, most engaging, self-contained moments suitable for short-form vertical clips.
Each clip MUST:
- Start and end on natural sentence/thought boundaries from the transcript.
- Have a duration between {min_clip_dur:.1f} seconds and {max_clip_dur:.1f} seconds (and NEVER exceed {video_duration:.2f} seconds).
- Have strong hook value, clear standalone context, and high storytelling or informational payoff.
- Support English, Hindi, and Hinglish naturally.

TRANSCRIPT:
{transcript_block}

Return strictly valid JSON matching this schema:
{{
  "moments": [
    {{
      "title": "Concise, punchy clip title (in the video's language or English)",
      "start_time": 0.0,
      "end_time": 15.2,
      "score": 92,
      "hook_strength": 90,
      "storytelling_score": 88,
      "standalone_value": 95,
      "language": "en | hi | hinglish",
      "rationale": "Specific explanation of why this moment works as a standalone short clip (hook, narrative arc, clarity)."
    }}
  ]
}}"""

    contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
    parsed, model_used = call_gemini_generate_content(
        contents=contents,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        preferred_model=preferred_model,
        temperature=0.2,
        timeout_sec=90,
    )

    raw_moments = parsed.get("moments", []) if isinstance(parsed, dict) else []
    validated_moments: List[BestMoment] = []

    for idx, item in enumerate(raw_moments):
        try:
            raw_start = float(item.get("start_time", 0.0))
            raw_end = float(item.get("end_time", 0.0))
        except (TypeError, ValueError):
            continue

        start_time = max(0.0, min(raw_start, video_duration - 1.0))
        end_time = max(start_time + 1.0, min(raw_end, video_duration))

        # Snap start_time and end_time to nearest transcript segment boundaries if within 1.2s
        for seg in transcript.segments:
            if abs(seg.start - start_time) <= 1.2:
                start_time = seg.start
            if abs(seg.end - end_time) <= 1.2:
                end_time = seg.end

        start_time = round(max(0.0, start_time), 2)
        end_time = round(min(video_duration, max(start_time + 1.0, end_time)), 2)
        clip_dur = round(end_time - start_time, 2)

        excerpt_parts = [
            s.text
            for s in transcript.segments
            if s.end > start_time + 0.1 and s.start < end_time - 0.1
        ]
        excerpt = " ".join(excerpt_parts).strip() or transcript.full_text[:160]

        score = max(1, min(100, int(item.get("score", 85))))
        hook = max(1, min(100, int(item.get("hook_strength", score))))
        story = max(1, min(100, int(item.get("storytelling_score", score))))
        standalone = max(1, min(100, int(item.get("standalone_value", score))))

        validated_moments.append(
            BestMoment(
                clip_id=f"clip_{idx + 1}",
                title=str(item.get("title") or f"Highlight Clip #{idx + 1}").strip(),
                start_time=start_time,
                end_time=end_time,
                duration=clip_dur,
                score=score,
                hook_strength=hook,
                storytelling_score=story,
                standalone_value=standalone,
                language=str(item.get("language") or transcript.language),
                rationale=str(
                    item.get("rationale")
                    or "Self-contained segment with high vocal engagement and clear narrative payoff."
                ).strip(),
                transcript_excerpt=excerpt,
                selected=True,
            )
        )

    validated_moments.sort(key=lambda m: m.score, reverse=True)
    return validated_moments, model_used
