from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING_MEDIA = "extracting_media"
    TRANSCRIBING = "transcribing"
    ANALYZING_MOMENTS = "analyzing_moments"
    READY_FOR_CLIPPING = "ready_for_clipping"
    RENDERING_CLIPS = "rendering_clips"
    COMPLETED = "completed"
    FAILED = "failed"


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class TranscriptSegment(BaseModel):
    id: int
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    language: str = Field(default="en")
    words: List[WordTimestamp] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    language: str = Field(default="en", description="Detected primary language (en, hi, hinglish)")
    duration: float = Field(..., ge=0.0)
    full_text: str
    provider: str = Field(default="gemini_multimodal_audio")
    segments: List[TranscriptSegment] = Field(default_factory=list)


class BestMoment(BaseModel):
    clip_id: str
    title: str
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    score: int = Field(..., ge=1, le=100)
    hook_strength: int = Field(default=80, ge=1, le=100)
    storytelling_score: int = Field(default=80, ge=1, le=100)
    standalone_value: int = Field(default=80, ge=1, le=100)
    language: str = Field(default="en")
    rationale: str
    transcript_excerpt: str
    selected: bool = Field(default=True)


class FramingKeyframe(BaseModel):
    timestamp: float
    center_x_norm: float = Field(..., ge=0.0, le=1.0)
    detection_method: str


class GeneratedClip(BaseModel):
    clip_id: str
    title: str
    start_time: float
    end_time: float
    duration: float
    width: int
    height: int
    aspect_ratio: str = "9:16"
    file_name: str
    file_size_bytes: int
    download_url: str
    stream_url: str
    framing_mode: str
    framing_summary: str
    caption_word_count: int
    has_audio: bool
    verified_playable: bool


class VideoMetadata(BaseModel):
    source_type: str  # "upload" or "youtube"
    original_name: str
    source_url: Optional[str] = None
    video_path: str
    audio_path: Optional[str] = None
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str] = None
    has_audio: bool
    file_size_bytes: int


class JobRecord(BaseModel):
    job_id: str
    created_at: str
    updated_at: str
    status: JobStatus
    progress_pct: int = 0
    step_message: str = "Initialized"
    error_message: Optional[str] = None
    gemini_model_used: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    transcript: Optional[TranscriptResult] = None
    moments: List[BestMoment] = Field(default_factory=list)
    clips: List[GeneratedClip] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)


class YouTubeIngestRequest(BaseModel):
    url: str = Field(..., min_length=5)
    gemini_model: Optional[str] = None
    language_hint: Optional[str] = "auto"
    auto_render_top_clips: bool = False


class ClipRenderItem(BaseModel):
    clip_id: str
    title: str = "Clip"
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    burn_captions: bool = True
    subject_tracking: bool = True


class RenderClipsRequest(BaseModel):
    clips: List[ClipRenderItem]
    target_resolution: str = Field(default="1080x1920")


class ApiKeyConfigRequest(BaseModel):
    gemini_api_key: str
    preferred_model: Optional[str] = "gemini-2.5-flash"
