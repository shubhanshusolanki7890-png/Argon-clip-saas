# ArgonClip — AI Vertical Video Clipping SaaS

ArgonClip is a full-stack AI video clipping application that converts long-form videos (direct video uploads or YouTube URLs) into downloadable **9:16 vertical MP4 clips** with **OpenCV subject-aware framing** and **burned-in animated word-level captions** (white normal words, yellow active spoken word).

Strictly configured to use **Google AI Studio Free-Tier Gemini Models** (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`) with automatic rate-limit backoff and model fallback.

## Workflow Architecture

`Video URL / Upload` → `FFprobe Validation & 16kHz WAV Audio Extraction` → `Multimodal Transcription & Acoustic RMS Word Timestamp Alignment (Hindi / English / Hinglish)` → `Gemini Best-Moment Detection & Timestamp Boundary Snapping` → `OpenCV Haar Face + Motion Saliency 9:16 Framing` → `ASS Animated White/Yellow Word Captions` → `FFmpeg H.264/AAC MP4 Render & Download`

## Project Structure

```text
argon-clip-saas/
├── backend/
│   ├── config.py               # Environment config & Free-Tier Gemini model whitelist
│   ├── main.py                 # FastAPI server, CORS, background jobs, streaming & downloads
│   ├── models.py               # Pydantic schemas for jobs, transcripts, moments, and clips
│   └── services/
│       ├── video_input.py      # Direct upload & yt-dlp YouTube ingestion + ffprobe validation
│       ├── gemini_service.py   # Gemini Free-Tier REST integration, chunked audio & fallback chain
│       ├── transcription.py    # Multilingual transcription + WAV RMS energy word alignment
│       ├── framing.py          # OpenCV Haar face + Sobel motion saliency 9:16 crop tracker
│       ├── captions.py         # ASS subtitle generator (white normal text, yellow active word)
│       ├── clipper.py          # FFmpeg 9:16 rendering & post-export ffprobe verification
│       └── job_store.py        # Thread-safe persistent JSON background job state store
├── frontend/
│   ├── package.json            # Frontend build configuration
│   ├── build.mjs               # Zero-dependency Node production bundler -> frontend/dist/
│   └── src/
│       ├── index.html          # Main application UI
│       ├── styles.css          # Responsive dark UI styles
│       └── app.js              # API client, live job polling, clip editor & MP4 player
├── tests/
│   └── test_e2e_pipeline.py    # End-to-end automated pipeline & MP4 verification test suite
├── Dockerfile                  # Production container image (Python 3.11, FFmpeg, libass, Node)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── run.sh                      # One-command build & launch script
```

## Required Environment Variables

Copy `.env.example` to `.env`:

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Yes | *(empty)* | Google AI Studio API key (can also be set at runtime via the UI or `POST /api/config/key`) |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Free-tier Gemini model (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`) |
| `PORT` | No | `8000` | HTTP server port |
| `STORAGE_DIR` | No | `./storage` | Persistent directory for uploads, jobs, and exported 9:16 MP4 clips |
| `MAX_UPLOAD_SIZE_MB` | No | `500` | Maximum video upload size in MB |

## Quick Start (Local / VM / Google Colab)

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg fonts-dejavu-core fonts-noto-core nodejs npm

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set your Free-Tier Gemini API Key
export GEMINI_API_KEY="your_api_key_here"

# 4. Build frontend and start FastAPI backend
chmod +x run.sh
./run.sh
```

Open `http://localhost:8000` in your browser.
