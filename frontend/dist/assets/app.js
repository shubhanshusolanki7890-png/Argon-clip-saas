const API_BASE = window.location.origin;

let activeJobId = null;
let pollTimer = null;
let currentJobData = null;

// DOM Elements
const pillFfmpeg = document.getElementById('pillFfmpeg');
const pillGemini = document.getElementById('pillGemini');
const pillModel = document.getElementById('pillModel');

const apiKeyInput = document.getElementById('apiKeyInput');
const freeModelSelect = document.getElementById('freeModelSelect');
const saveKeyBtn = document.getElementById('saveKeyBtn');
const testKeyBtn = document.getElementById('testKeyBtn');
const configFeedback = document.getElementById('configFeedback');

const tabUploadBtn = document.getElementById('tabUploadBtn');
const tabYoutubeBtn = document.getElementById('tabYoutubeBtn');
const uploadForm = document.getElementById('uploadForm');
const youtubeForm = document.getElementById('youtubeForm');
const ingestFeedback = document.getElementById('ingestFeedback');

const jobsListContainer = document.getElementById('jobsListContainer');
const refreshJobsBtn = document.getElementById('refreshJobsBtn');

const jobStatusBadge = document.getElementById('jobStatusBadge');
const jobProgressBar = document.getElementById('jobProgressBar');
const jobStepText = document.getElementById('jobStepText');
const jobErrorBox = document.getElementById('jobErrorBox');
const videoMetaBox = document.getElementById('videoMetaBox');
const jobLogsPre = document.getElementById('jobLogsPre');

const momentsListContainer = document.getElementById('momentsListContainer');
const addCustomMomentBtn = document.getElementById('addCustomMomentBtn');
const renderSelectedBtn = document.getElementById('renderSelectedBtn');
const targetResSelect = document.getElementById('targetResSelect');
const optSubjectTracking = document.getElementById('optSubjectTracking');
const optBurnCaptions = document.getElementById('optBurnCaptions');

const clipsGalleryContainer = document.getElementById('clipsGalleryContainer');
const clipCountBadge = document.getElementById('clipCountBadge');

const transcriptContainer = document.getElementById('transcriptContainer');
const transcriptLangBadge = document.getElementById('transcriptLangBadge');

function showFeedback(el, message, type = 'info') {
  el.textContent = message;
  el.className = `feedback-box ${type}`;
  el.classList.remove('hidden');
}

function hideFeedback(el) {
  el.classList.add('hidden');
}

async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    pillFfmpeg.textContent = data.ffmpeg_available ? 'FFmpeg 9:16: Ready' : 'FFmpeg: Missing';
    pillFfmpeg.className = `pill ${data.ffmpeg_available ? 'ok' : 'warn'}`;

    pillGemini.textContent = data.gemini_api_key_configured
      ? 'Gemini API Key: Configured'
      : 'Gemini API Key: Not Set';
    pillGemini.className = `pill ${data.gemini_api_key_configured ? 'ok' : 'warn'}`;

    pillModel.textContent = `Free Model: ${data.active_gemini_model}`;
    if (data.active_gemini_model) {
      freeModelSelect.value = data.active_gemini_model;
    }
  } catch (err) {
    pillFfmpeg.textContent = 'Backend: Offline';
    pillFfmpeg.className = 'pill warn';
  }
}

saveKeyBtn.addEventListener('click', async () => {
  const key = apiKeyInput.value.trim();
  const model = freeModelSelect.value;
  if (!key) {
    showFeedback(configFeedback, 'Please enter a valid Gemini API key.', 'error');
    return;
  }
  saveKeyBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/config/key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gemini_api_key: key, preferred_model: model }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to save API key');
    apiKeyInput.value = '';
    showFeedback(
      configFeedback,
      `Saved key on backend server. Active free model: ${data.preferred_model}`,
      'success'
    );
    await fetchHealth();
  } catch (err) {
    showFeedback(configFeedback, err.message, 'error');
  } finally {
    saveKeyBtn.disabled = false;
  }
});

testKeyBtn.addEventListener('click', async () => {
  testKeyBtn.disabled = true;
  showFeedback(configFeedback, 'Testing connectivity with free-tier Gemini API...', 'info');
  try {
    const model = encodeURIComponent(freeModelSelect.value);
    const res = await fetch(`${API_BASE}/api/config/test-gemini?preferred_model=${model}`, {
      method: 'POST',
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Gemini connectivity test failed');
    showFeedback(
      configFeedback,
      `Connected to Gemini Free Tier (${data.model_used}). Status: Ready.`,
      'success'
    );
  } catch (err) {
    showFeedback(configFeedback, `Gemini API Error: ${err.message}`, 'error');
  } finally {
    testKeyBtn.disabled = false;
  }
});

// Tab switching
tabUploadBtn.addEventListener('click', () => {
  tabUploadBtn.classList.add('active');
  tabYoutubeBtn.classList.remove('active');
  uploadForm.classList.remove('hidden');
  youtubeForm.classList.add('hidden');
});

tabYoutubeBtn.addEventListener('click', () => {
  tabYoutubeBtn.classList.add('active');
  tabUploadBtn.classList.remove('active');
  youtubeForm.classList.remove('hidden');
  uploadForm.classList.add('hidden');
});

// Direct File Upload Submission
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('videoFileInput');
  const subInput = document.getElementById('subtitleFileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    showFeedback(ingestFeedback, 'Please select a video file to upload.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('video_file', fileInput.files[0]);
  if (subInput.files && subInput.files.length > 0) {
    formData.append('subtitle_file', subInput.files[0]);
  }
  formData.append('language_hint', document.getElementById('uploadLangSelect').value);
  formData.append('gemini_model', freeModelSelect.value);
  formData.append('auto_render_top_clips', document.getElementById('uploadAutoRender').checked);

  const btn = document.getElementById('uploadSubmitBtn');
  btn.disabled = true;
  showFeedback(ingestFeedback, 'Uploading video and starting analysis job...', 'info');

  try {
    const res = await fetch(`${API_BASE}/api/jobs/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    hideFeedback(ingestFeedback);
    selectActiveJob(data.job_id);
    await loadRecentJobs();
  } catch (err) {
    showFeedback(ingestFeedback, err.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

// YouTube URL Submission
youtubeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const urlVal = document.getElementById('youtubeUrlInput').value.trim();
  if (!urlVal) {
    showFeedback(ingestFeedback, 'Please enter a valid YouTube URL.', 'error');
    return;
  }

  const btn = document.getElementById('ytSubmitBtn');
  btn.disabled = true;
  showFeedback(ingestFeedback, 'Submitting YouTube extraction job...', 'info');

  try {
    const res = await fetch(`${API_BASE}/api/jobs/youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: urlVal,
        language_hint: document.getElementById('ytLangSelect').value,
        gemini_model: freeModelSelect.value,
        auto_render_top_clips: document.getElementById('ytAutoRender').checked,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'YouTube job creation failed');
    hideFeedback(ingestFeedback);
    selectActiveJob(data.job_id);
    await loadRecentJobs();
  } catch (err) {
    showFeedback(ingestFeedback, err.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

async function loadRecentJobs() {
  try {
    const res = await fetch(`${API_BASE}/api/jobs`);
    const data = await res.json();
    const jobs = data.jobs || [];
    if (jobs.length === 0) {
      jobsListContainer.innerHTML = '<p class="muted">No jobs submitted yet.</p>';
      return;
    }
    jobsListContainer.innerHTML = '';
    jobs.forEach((j) => {
      const row = document.createElement('div');
      row.className = 'job-row';
      const title = j.metadata ? j.metadata.original_name : j.job_id;
      row.innerHTML = `
        <span><strong>${escapeHtml(title)}</strong> <small class="muted">(${j.job_id})</small></span>
        <span class="status-badge">${escapeHtml(j.status)}</span>
      `;
      row.addEventListener('click', () => selectActiveJob(j.job_id));
      jobsListContainer.appendChild(row);
    });
  } catch (err) {
    console.error('Failed to load jobs:', err);
  }
}

refreshJobsBtn.addEventListener('click', loadRecentJobs);

function selectActiveJob(jobId) {
  activeJobId = jobId;
  if (pollTimer) clearInterval(pollTimer);
  pollJobStatus();
  pollTimer = setInterval(pollJobStatus, 2000);
}

async function pollJobStatus() {
  if (!activeJobId) return;
  try {
    const res = await fetch(`${API_BASE}/api/jobs/${activeJobId}`);
    if (!res.ok) return;
    const job = await res.json();
    currentJobData = job;
    renderJobState(job);

    if (['completed', 'failed', 'ready_for_clipping'].includes(job.status)) {
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      loadRecentJobs();
    }
  } catch (err) {
    console.error('Polling error:', err);
  }
}

function renderJobState(job) {
  jobStatusBadge.textContent = job.status;
  jobProgressBar.style.width = `${job.progress_pct || 0}%`;
  jobStepText.textContent = job.step_message || '';

  if (job.error_message) {
    showFeedback(jobErrorBox, job.error_message, 'error');
  } else {
    hideFeedback(jobErrorBox);
  }

  jobLogsPre.textContent = (job.logs || []).join('\n');

  if (job.metadata) {
    const m = job.metadata;
    videoMetaBox.classList.remove('hidden');
    videoMetaBox.innerHTML = `
      <div><strong>Source:</strong> ${escapeHtml(m.original_name)}</div>
      <div><strong>Duration:</strong> ${m.duration.toFixed(2)}s</div>
      <div><strong>Resolution:</strong> ${m.width}x${m.height} (${m.fps} fps)</div>
      <div><strong>Codecs:</strong> ${escapeHtml(m.video_codec)} / ${escapeHtml(m.audio_codec || 'none')}</div>
      <div><strong>Size:</strong> ${(m.file_size_bytes / (1024 * 1024)).toFixed(2)} MB</div>
      <div><strong>Gemini Model:</strong> ${escapeHtml(job.gemini_model_used || 'pending')}</div>
    `;
    addCustomMomentBtn.disabled = false;
  } else {
    videoMetaBox.classList.add('hidden');
    addCustomMomentBtn.disabled = true;
  }

  renderMomentsList(job);
  renderClipsGallery(job);
  renderTranscript(job);
}

function renderMomentsList(job) {
  const moments = job.moments || [];
  const maxDur = job.metadata ? job.metadata.duration : 3600;

  if (moments.length === 0) {
    momentsListContainer.innerHTML =
      '<p class="muted">Moments will appear here once transcription and Gemini analysis finish.</p>';
    renderSelectedBtn.disabled = true;
    return;
  }

  // Avoid overwriting user input while editing if the same job and moment count are already rendered
  const existingCards = momentsListContainer.querySelectorAll('.moment-item');
  if (
    existingCards.length === moments.length &&
    momentsListContainer.dataset.jobId === job.job_id &&
    job.status === 'rendering_clips'
  ) {
    return;
  }

  momentsListContainer.dataset.jobId = job.job_id;
  momentsListContainer.innerHTML = '';

  moments.forEach((m, idx) => {
    const div = document.createElement('div');
    div.className = 'moment-item';
    div.dataset.clipId = m.clip_id;
    div.innerHTML = `
      <div class="moment-top">
        <label class="moment-title-wrap">
          <input type="checkbox" class="moment-check" data-idx="${idx}" ${m.selected ? 'checked' : ''} />
          <input type="text" class="moment-title-input" value="${escapeHtml(m.title)}" style="width: 260px; padding: 0.25rem 0.5rem;" />
        </label>
        <span class="score-pill">Score: ${m.score}/100 (Hook ${m.hook_strength})</span>
      </div>
      <p class="moment-rationale">${escapeHtml(m.rationale)}</p>
      <div class="moment-excerpt">"${escapeHtml(m.transcript_excerpt)}"</div>
      <div class="moment-time-controls">
        <label>Start (s): <input type="number" class="moment-start" step="0.1" min="0" max="${maxDur}" value="${m.start_time}" /></label>
        <label>End (s): <input type="number" class="moment-end" step="0.1" min="0.5" max="${maxDur}" value="${m.end_time}" /></label>
        <span class="muted">Source Bound: 0.0s – ${maxDur.toFixed(2)}s</span>
      </div>
    `;
    momentsListContainer.appendChild(div);
  });

  renderSelectedBtn.disabled = job.status === 'rendering_clips';
}

addCustomMomentBtn.addEventListener('click', () => {
  if (!currentJobData || !currentJobData.metadata) return;
  const maxDur = currentJobData.metadata.duration;
  const nextIdx = (currentJobData.moments || []).length + 1;
  const customMoment = {
    clip_id: `custom_${nextIdx}`,
    title: `Custom Highlight #${nextIdx}`,
    start_time: 0.0,
    end_time: Math.min(15.0, maxDur),
    duration: Math.min(15.0, maxDur),
    score: 90,
    hook_strength: 90,
    storytelling_score: 90,
    standalone_value: 90,
    language: currentJobData.transcript ? currentJobData.transcript.language : 'en',
    rationale: 'User-defined custom timestamp range.',
    transcript_excerpt: 'Custom selected clip window.',
    selected: true,
  };
  currentJobData.moments = currentJobData.moments || [];
  currentJobData.moments.push(customMoment);
  momentsListContainer.dataset.jobId = '';
  renderMomentsList(currentJobData);
});

renderSelectedBtn.addEventListener('click', async () => {
  if (!activeJobId || !currentJobData) return;
  const items = momentsListContainer.querySelectorAll('.moment-item');
  const selectedClips = [];

  items.forEach((itemEl) => {
    const check = itemEl.querySelector('.moment-check');
    if (check && check.checked) {
      const clipId = itemEl.dataset.clipId;
      const title = itemEl.querySelector('.moment-title-input').value.trim() || 'Clip';
      const startVal = parseFloat(itemEl.querySelector('.moment-start').value);
      const endVal = parseFloat(itemEl.querySelector('.moment-end').value);
      selectedClips.push({
        clip_id: clipId,
        title,
        start_time: startVal,
        end_time: endVal,
        burn_captions: optBurnCaptions.checked,
        subject_tracking: optSubjectTracking.checked,
      });
    }
  });

  if (selectedClips.length === 0) {
    alert('Please select at least one moment to generate a 9:16 vertical clip.');
    return;
  }

  renderSelectedBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/api/jobs/${activeJobId}/render`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        clips: selectedClips,
        target_resolution: targetResSelect.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to start clip rendering');
    selectActiveJob(activeJobId);
  } catch (err) {
    alert(`Render Error: ${err.message}`);
    renderSelectedBtn.disabled = false;
  }
});

function renderClipsGallery(job) {
  const clips = job.clips || [];
  clipCountBadge.textContent = `${clips.length} clip(s)`;
  if (clips.length === 0) {
    clipsGalleryContainer.innerHTML =
      '<p class="muted">Rendered 9:16 vertical MP4 clips will appear here with preview player and direct download links.</p>';
    return;
  }

  clipsGalleryContainer.innerHTML = '';
  clips.forEach((c) => {
    const card = document.createElement('div');
    card.className = 'clip-card';
    const sizeMb = (c.file_size_bytes / (1024 * 1024)).toFixed(2);
    card.innerHTML = `
      <div>
        <strong>${escapeHtml(c.title)}</strong>
        <div class="clip-meta">${c.start_time.toFixed(1)}s – ${c.end_time.toFixed(1)}s (${c.duration.toFixed(1)}s) • ${c.width}x${c.height} (${c.aspect_ratio})</div>
      </div>
      <div class="clip-video-wrap">
        <video controls playsinline preload="metadata" src="${API_BASE}${c.stream_url}"></video>
      </div>
      <div class="clip-meta">
        <div><strong>Framing:</strong> ${escapeHtml(c.framing_summary)}</div>
        <div><strong>Animated Word Captions:</strong> ${c.caption_word_count} words (White/Yellow ASS)</div>
        <div><strong>Verified MP4:</strong> ${sizeMb} MB • Audio: ${c.has_audio ? 'AAC Stereo' : 'No'}</div>
      </div>
      <a class="btn btn-primary btn-block" style="text-align:center; text-decoration:none;" href="${API_BASE}${c.download_url}" download="${escapeHtml(c.file_name)}">
        Download 9:16 MP4 (${sizeMb} MB)
      </a>
    `;
    clipsGalleryContainer.appendChild(card);
  });
}

function renderTranscript(job) {
  const t = job.transcript;
  if (!t || !t.segments || t.segments.length === 0) {
    transcriptLangBadge.textContent = 'Language: -';
    transcriptContainer.innerHTML =
      '<p class="muted">Full transcript and word-level timing markers will appear here.</p>';
    return;
  }

  transcriptLangBadge.textContent = `Language: ${t.language.toUpperCase()} • Provider: ${t.provider}`;
  transcriptContainer.innerHTML = '';

  t.segments.forEach((seg) => {
    const row = document.createElement('div');
    row.className = 'seg-row';
    const wordsHtml = (seg.words || [])
      .map(
        (w) =>
          `<span class="word-chip" title="${w.start.toFixed(2)}s - ${w.end.toFixed(2)}s">${escapeHtml(w.word)} <small>(${w.start.toFixed(1)}s)</small></span>`
      )
      .join('');
    row.innerHTML = `
      <div>
        <span class="seg-time">[${seg.start.toFixed(2)}s – ${seg.end.toFixed(2)}s]</span>
        <span>${escapeHtml(seg.text)}</span>
      </div>
      <div class="word-chips">${wordsHtml}</div>
    `;
    transcriptContainer.appendChild(row);
  });
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Initial boot
fetchHealth();
loadRecentJobs();
