import re
from pathlib import Path
from typing import List, Tuple

from backend.models import TranscriptSegment, WordTimestamp

DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")


def _format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    s = max(0.0, float(seconds))
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    centis = int(round((s - int(s)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(word: str) -> str:
    """Sanitize word text so ASS override tags are never corrupted."""
    return word.replace("\\", "").replace("{", "(").replace("}", ")").strip()


def extract_clip_words(
    segments: List[TranscriptSegment],
    clip_start: float,
    clip_end: float,
) -> List[WordTimestamp]:
    """
    Extract all word timestamps overlapping [clip_start, clip_end] and shift their
    coordinates to clip-relative time [0.0, clip_duration].
    """
    clip_duration = max(0.1, clip_end - clip_start)
    rel_words: List[WordTimestamp] = []

    for seg in segments:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        seg_words = seg.words
        if not seg_words:
            # Fallback if segment had text but empty words list
            tokens = [t for t in seg.text.split() if t.strip()]
            if not tokens:
                continue
            dur = max(0.1, seg.end - seg.start)
            step = dur / len(tokens)
            seg_words = [
                WordTimestamp(
                    word=tok,
                    start=seg.start + i * step,
                    end=seg.start + (i + 1) * step,
                )
                for i, tok in enumerate(tokens)
            ]

        for w in seg_words:
            if w.end <= clip_start + 0.02 or w.start >= clip_end - 0.02:
                continue
            r_start = max(0.0, w.start - clip_start)
            r_end = min(clip_duration, max(r_start + 0.05, w.end - clip_start))
            rel_words.append(
                WordTimestamp(
                    word=w.word,
                    start=round(r_start, 3),
                    end=round(r_end, 3),
                    confidence=w.confidence,
                )
            )

    return rel_words


def generate_animated_ass_captions(
    segments: List[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    output_ass_path: Path,
    target_width: int = 1080,
    target_height: int = 1920,
    words_per_group: int = 4,
) -> Tuple[Path, int]:
    """
    Generate an Advanced SubStation Alpha (.ass) subtitle file for a vertical clip:
    - Normal words appear in crisp WHITE (&H00FFFFFF).
    - The currently spoken word is highlighted in vivid YELLOW (&H0000FFFF in ASS BGR = #FFFF00 RGB)
      with a subtle 108% scale emphasis.
    - Positioned near the bottom safe zone (Alignment=2, MarginV scaled to resolution).
    - Supports English, Hindi, and Hinglish.
    Returns (output_ass_path, total_word_count).
    """
    clip_words = extract_clip_words(segments, clip_start, clip_end)
    all_text = " ".join(w.word for w in clip_words)
    font_name = "Noto Sans" if DEVANAGARI_REGEX.search(all_text) else "DejaVu Sans"

    scale_factor = float(target_height) / 1920.0
    font_size = max(44, int(round(68 * scale_factor)))
    outline_px = max(3, int(round(5 * scale_factor)))
    shadow_px = max(2, int(round(3 * scale_factor)))
    margin_v = max(140, int(round(250 * scale_factor)))
    margin_h = max(40, int(round(70 * scale_factor)))

    # ASS Color Format: &HAABBGGRR (White = &H00FFFFFF, Yellow RGB #FFFF00 = BGR &H0000FFFF, Black = &H00000000)
    ass_header = f"""[Script Info]
Title: ArgonClip Animated Word Captions
ScriptType: v4.00+
PlayResX: {target_width}
PlayResY: {target_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ArgonCaption,{font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,{outline_px},{shadow_px},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: List[str] = []

    # Group words into readable short phrases (3 to 4 words per group, or split on long pauses > 0.65s)
    groups: List[List[WordTimestamp]] = []
    current_group: List[WordTimestamp] = []

    for w in clip_words:
        if current_group:
            gap = w.start - current_group[-1].end
            if len(current_group) >= words_per_group or gap > 0.65:
                groups.append(current_group)
                current_group = []
        current_group.append(w)
    if current_group:
        groups.append(current_group)

    for group in groups:
        for active_idx, active_word in enumerate(group):
            ev_start = active_word.start
            # Extend event end to next word's start inside the same group for seamless display
            if active_idx + 1 < len(group):
                ev_end = max(ev_start + 0.05, group[active_idx + 1].start)
            else:
                ev_end = max(ev_start + 0.08, active_word.end)

            start_str = _format_ass_time(ev_start)
            end_str = _format_ass_time(ev_end)

            rendered_tokens: List[str] = []
            for idx, token_obj in enumerate(group):
                safe_word = _escape_ass_text(token_obj.word)
                if not safe_word:
                    continue
                if idx == active_idx:
                    # Active word: Yellow (&H0000FFFF&) + slight scale pop, then reset to White (&H00FFFFFF&)
                    rendered_tokens.append(
                        f"{{\\1c&H0000FFFF&\\fscx108\\fscy108}}{safe_word}{{\\1c&H00FFFFFF&\\fscx100\\fscy100}}"
                    )
                else:
                    # Normal word: White (&H00FFFFFF&)
                    rendered_tokens.append(f"{{\\1c&H00FFFFFF&}}{safe_word}")

            line_text = " ".join(rendered_tokens)
            dialogue_lines.append(
                f"Dialogue: 0,{start_str},{end_str},ArgonCaption,,0,0,0,,{line_text}"
            )

    output_ass_path.write_text(ass_header + "\n".join(dialogue_lines) + "\n", encoding="utf-8")
    return output_ass_path, len(clip_words)
