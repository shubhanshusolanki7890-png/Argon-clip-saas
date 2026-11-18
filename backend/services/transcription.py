import re
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from backend.models import TranscriptResult, TranscriptSegment, WordTimestamp
from backend.services.gemini_service import (
    GeminiServiceError,
    transcribe_audio_chunk_with_gemini,
)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
HINDI_ROMAN_MARKERS = {
    "hai",
    "hain",
    "nahi",
    "kya",
    "aur",
    "mein",
    "apne",
    "aap",
    "karna",
    "kaise",
    "bahut",
    "lekin",
    "agar",
    "toh",
    "bhi",
    "accha",
    "dekho",
    "bhai",
    "dosto",
    "yaar",
    "kyunki",
    "iska",
    "uska",
    "hum",
    "tum",
    "yeh",
    "woh",
    "abhi",
    "sahi",
    "paisa",
    "kaam",
}


def detect_language_tag(text: str) -> str:
    """Detect whether text is Hindi (Devanagari), Hinglish (mixed Hindi-English), or English."""
    if not text.strip():
        return "en"
    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_devanagari and has_latin:
        return "hinglish"
    if has_devanagari:
        return "hi"
    tokens = {re.sub(r"[^\w]", "", w.lower()) for w in text.split()}
    hindi_hits = len(tokens.intersection(HINDI_ROMAN_MARKERS))
    if hindi_hits >= 2 or (len(tokens) > 0 and hindi_hits / max(1, len(tokens)) >= 0.2):
        return "hinglish"
    return "en"


def load_wav_energy_envelope(wav_path: Path, hop_ms: int = 20) -> Tuple[np.ndarray, float]:
    """
    Compute short-time RMS energy envelope of a 16kHz mono WAV file at 20ms resolution.
    Used to snap and refine word-level timestamps to actual acoustic speech energy bursts.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return np.ones(100, dtype=np.float32), 0.02

    hop_samples = max(1, int(sr * (hop_ms / 1000.0)))
    n_hops = max(1, len(samples) // hop_samples)
    trimmed = samples[: n_hops * hop_samples].reshape(n_hops, hop_samples)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-9)
    return rms, hop_ms / 1000.0


def align_words_within_segment(
    text: str,
    seg_start: float,
    seg_end: float,
    raw_words: Optional[List[dict]] = None,
    rms_envelope: Optional[np.ndarray] = None,
    hop_sec: float = 0.02,
) -> List[WordTimestamp]:
    """
    Produce monotonically non-overlapping word-level timestamps inside [seg_start, seg_end].
    If raw word timestamps are provided and valid, validates & clamps them; otherwise distributes
    word durations weighted by character length and refines boundaries using the audio RMS envelope.
    """
    tokens = [t.strip() for t in text.strip().split() if t.strip()]
    if not tokens:
        return []

    seg_dur = max(0.15, seg_end - seg_start)

    # Check if raw_words from provider match token count and lie inside segment bounds
    if raw_words and len(raw_words) == len(tokens):
        aligned: List[WordTimestamp] = []
        cursor = seg_start
        valid_raw = True
        for tok, rw in zip(tokens, raw_words):
            try:
                w_start = float(rw.get("start", cursor))
                w_end = float(rw.get("end", w_start + 0.2))
            except (TypeError, ValueError):
                valid_raw = False
                break
            w_start = max(cursor, min(w_start, seg_end - 0.04))
            w_end = max(w_start + 0.04, min(w_end, seg_end))
            aligned.append(
                WordTimestamp(
                    word=tok,
                    start=round(w_start, 3),
                    end=round(w_end, 3),
                    confidence=float(rw.get("confidence", 0.95)),
                )
            )
            cursor = w_end
        if valid_raw and aligned:
            return aligned

    # Trim leading/trailing silence inside segment using acoustic RMS envelope if available
    active_start = seg_start
    active_end = seg_end
    if rms_envelope is not None and len(rms_envelope) > 0:
        idx_s = max(0, min(len(rms_envelope) - 1, int(seg_start / hop_sec)))
        idx_e = max(idx_s + 1, min(len(rms_envelope), int(seg_end / hop_sec)))
        sub_rms = rms_envelope[idx_s:idx_e]
        if len(sub_rms) > 4:
            thresh = float(np.percentile(sub_rms, 20)) + 0.12 * float(
                np.max(sub_rms) - np.min(sub_rms)
            )
            voiced = np.where(sub_rms >= thresh)[0]
            if len(voiced) >= 2:
                v_start = seg_start + float(voiced[0]) * hop_sec
                v_end = seg_start + float(voiced[-1] + 1) * hop_sec
                if v_end - v_start >= min(0.25, seg_dur * 0.4):
                    active_start = max(seg_start, v_start)
                    active_end = min(seg_end, v_end)

    active_dur = max(0.12, active_end - active_start)
    weights = np.array([max(2, len(re.sub(r"[^\w]", "", w))) for w in tokens], dtype=np.float32)
    # Add slight pause weight after punctuation
    for i, w in enumerate(tokens):
        if w.endswith((".", "!", "?", "।", ",", ";", ":")):
            weights[i] += 1.5
    weights /= np.sum(weights)

    words_out: List[WordTimestamp] = []
    cursor = active_start
    for i, (tok, weight) in enumerate(zip(tokens, weights)):
        w_dur = float(weight) * active_dur
        w_start = cursor
        w_end = active_end if i == len(tokens) - 1 else min(active_end, w_start + max(0.05, w_dur))
        words_out.append(
            WordTimestamp(
                word=tok,
                start=round(w_start, 3),
                end=round(max(w_start + 0.04, w_end), 3),
                confidence=0.92,
            )
        )
        cursor = w_end

    return words_out


def parse_subtitle_file_to_transcript(
    sub_path: Path, wav_path: Path, duration: float
) -> TranscriptResult:
    """Parse an SRT or WebVTT file and align word-level timestamps against the audio envelope."""
    content = sub_path.read_text(encoding="utf-8", errors="replace")
    rms, hop_sec = load_wav_energy_envelope(wav_path)

    time_pattern = re.compile(
        r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})"
    )

    def _to_sec(h: Optional[str], m: str, s: str, ms: str) -> float:
        return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    blocks = re.split(r"\n\s*\n", content.strip())
    segments: List[TranscriptSegment] = []
    seg_id = 1

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines or lines[0].upper().startswith("WEBVTT"):
            continue
        match_line_idx = -1
        match_obj = None
        for idx, ln in enumerate(lines):
            m = time_pattern.search(ln)
            if m:
                match_line_idx = idx
                match_obj = m
                break
        if match_obj is None:
            continue

        g = match_obj.groups()
        start_t = max(0.0, min(_to_sec(g[0], g[1], g[2], g[3]), duration))
        end_t = max(start_t + 0.1, min(_to_sec(g[4], g[5], g[6], g[7]), duration))
        text_lines = lines[match_line_idx + 1 :]
        clean_text = re.sub(r"<[^>]+>", "", " ".join(text_lines)).strip()
        if not clean_text:
            continue

        lang = detect_language_tag(clean_text)
        words = align_words_within_segment(
            clean_text, start_t, end_t, raw_words=None, rms_envelope=rms, hop_sec=hop_sec
        )
        segments.append(
            TranscriptSegment(
                id=seg_id,
                start=round(start_t, 3),
                end=round(end_t, 3),
                text=clean_text,
                language=lang,
                words=words,
            )
        )
        seg_id += 1

    full_text = " ".join(s.text for s in segments)
    overall_lang = detect_language_tag(full_text)
    return TranscriptResult(
        language=overall_lang,
        duration=duration,
        full_text=full_text,
        provider="subtitle_track_aligned",
        segments=segments,
    )


def try_local_whisper_transcription(
    wav_path: Path, duration: float, language_hint: str = "auto"
) -> Optional[TranscriptResult]:
    """Attempt transcription with faster_whisper or openai-whisper if installed locally."""
    rms, hop_sec = load_wav_energy_envelope(wav_path)
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")
        lang_arg = None if language_hint in ("auto", "hinglish", "") else language_hint
        raw_segs, info = model.transcribe(str(wav_path), language=lang_arg, word_timestamps=True)
        segments: List[TranscriptSegment] = []
        for idx, s in enumerate(raw_segs, start=1):
            txt = s.text.strip()
            if not txt:
                continue
            raw_words = (
                [{"word": w.word.strip(), "start": w.start, "end": w.end} for w in (s.words or [])]
                if getattr(s, "words", None)
                else None
            )
            seg_lang = detect_language_tag(txt)
            words = align_words_within_segment(
                txt,
                max(0.0, float(s.start)),
                min(duration, float(s.end)),
                raw_words=raw_words,
                rms_envelope=rms,
                hop_sec=hop_sec,
            )
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=round(max(0.0, float(s.start)), 3),
                    end=round(min(duration, float(s.end)), 3),
                    text=txt,
                    language=seg_lang,
                    words=words,
                )
            )
        full_text = " ".join(seg.text for seg in segments)
        return TranscriptResult(
            language=detect_language_tag(full_text) or getattr(info, "language", "en"),
            duration=duration,
            full_text=full_text,
            provider="faster_whisper_word_level",
            segments=segments,
        )
    except ImportError:
        pass
    except Exception:
        pass

    return None


def transcribe_video_audio(
    wav_path: Path,
    duration: float,
    language_hint: str = "auto",
    subtitle_path: Optional[Path] = None,
    preferred_gemini_model: Optional[str] = None,
) -> Tuple[TranscriptResult, Optional[str]]:
    """
    End-to-end transcription pipeline supporting long videos, Hindi, English, and Hinglish:
    1. Uses local faster-whisper with word timestamps if installed.
    2. Uses verified subtitle track + acoustic WAV energy alignment if provided.
    3. Uses Gemini Free-Tier multimodal audio transcription in 600-second (10-min) chunks
       with acoustic WAV energy-envelope word-level timestamp alignment.
    """
    # 1. Check local faster-whisper
    whisper_res = try_local_whisper_transcription(wav_path, duration, language_hint)
    if whisper_res and whisper_res.segments:
        return whisper_res, None

    # 2. Check extracted subtitle file if available
    if subtitle_path and subtitle_path.exists():
        sub_res = parse_subtitle_file_to_transcript(subtitle_path, wav_path, duration)
        if sub_res.segments:
            return sub_res, None

    # 3. Gemini Free-Tier Multimodal Audio Transcription (chunked for long videos)
    rms, hop_sec = load_wav_energy_envelope(wav_path)
    chunk_size = 600.0  # 10 minutes per chunk for long-video reliability
    all_segments: List[TranscriptSegment] = []
    seg_id = 1
    model_used_final: Optional[str] = None
    chunk_start = 0.0

    while chunk_start < duration - 0.2:
        chunk_dur = min(chunk_size, duration - chunk_start)
        raw_segs, _, model_used = transcribe_audio_chunk_with_gemini(
            audio_wav_path=wav_path,
            chunk_start_sec=chunk_start,
            chunk_duration_sec=chunk_dur,
            language_hint=language_hint,
            preferred_model=preferred_gemini_model,
        )
        model_used_final = model_used

        for rs in raw_segs:
            txt = str(rs.get("text", "")).strip()
            if not txt:
                continue
            try:
                rel_s = max(0.0, float(rs.get("start", 0.0)))
                rel_e = max(rel_s + 0.15, float(rs.get("end", rel_s + 2.0)))
            except (TypeError, ValueError):
                continue

            abs_s = min(duration - 0.1, chunk_start + rel_s)
            abs_e = min(duration, chunk_start + rel_e)
            if abs_e <= abs_s:
                continue

            raw_words = rs.get("words") if isinstance(rs.get("words"), list) else None
            if raw_words:
                shifted_words = []
                for rw in raw_words:
                    if isinstance(rw, dict) and "word" in rw:
                        shifted_words.append(
                            {
                                "word": str(rw["word"]),
                                "start": chunk_start + float(rw.get("start", rel_s)),
                                "end": chunk_start + float(rw.get("end", rel_e)),
                            }
                        )
                raw_words = shifted_words

            words = align_words_within_segment(
                txt, abs_s, abs_e, raw_words=raw_words, rms_envelope=rms, hop_sec=hop_sec
            )
            seg_lang = detect_language_tag(txt)
            all_segments.append(
                TranscriptSegment(
                    id=seg_id,
                    start=round(abs_s, 3),
                    end=round(abs_e, 3),
                    text=txt,
                    language=seg_lang,
                    words=words,
                )
            )
            seg_id += 1

        chunk_start += chunk_size

    if not all_segments:
        raise GeminiServiceError(
            "Transcription completed but no spoken words were detected in the audio track.",
            status_code=422,
        )

    full_text = " ".join(s.text for s in all_segments)
    overall_lang = detect_language_tag(full_text)
    return (
        TranscriptResult(
            language=overall_lang,
            duration=duration,
            full_text=full_text,
            provider=f"gemini_multimodal ({model_used_final or 'gemini-2.5-flash'})",
            segments=all_segments,
        ),
        model_used_final,
    )
