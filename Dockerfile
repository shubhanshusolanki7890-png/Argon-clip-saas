FROM python:3.11-slim

# Install system dependencies: FFmpeg (with libass), OpenCV runtime libs, fonts for English & Hindi/Devanagari, Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    fonts-noto-core \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and build production frontend
COPY frontend ./frontend
RUN cd frontend && npm run build

COPY backend ./backend
RUN mkdir -p /app/storage/uploads /app/storage/clips /app/storage/temp /app/storage/jobs

ENV STORAGE_DIR=/app/storage
ENV GEMINI_MODEL=gemini-2.5-flash
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
