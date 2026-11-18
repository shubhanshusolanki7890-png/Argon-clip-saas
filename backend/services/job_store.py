from datetime import datetime, timezone
import json
import threading
from typing import Dict, List, Optional

from backend.config import JOBS_DIR
from backend.models import JobRecord, JobStatus

_LOCK = threading.RLock()
_JOBS_CACHE: Dict[str, JobRecord] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_job(job: JobRecord) -> str:
    if hasattr(job, "model_dump"):
        return json.dumps(job.model_dump(), indent=2)
    return json.dumps(job.dict(), indent=2)


def _deserialize_job(raw_json: str) -> JobRecord:
    if hasattr(JobRecord, "model_validate_json"):
        return JobRecord.model_validate_json(raw_json)
    return JobRecord.parse_raw(raw_json)


def save_job(job: JobRecord) -> JobRecord:
    with _LOCK:
        job.updated_at = _now_iso()
        _JOBS_CACHE[job.job_id] = job
        job_file = JOBS_DIR / f"{job.job_id}.json"
        job_file.write_text(_serialize_job(job), encoding="utf-8")
        return job


def create_job(job_id: str, step_message: str = "Job queued") -> JobRecord:
    now = _now_iso()
    record = JobRecord(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        status=JobStatus.QUEUED,
        progress_pct=5,
        step_message=step_message,
        logs=[f"[{now}] {step_message}"],
    )
    return save_job(record)


def update_job_progress(
    job_id: str,
    status: JobStatus,
    progress_pct: int,
    step_message: str,
    error_message: Optional[str] = None,
) -> Optional[JobRecord]:
    with _LOCK:
        job = get_job(job_id)
        if not job:
            return None
        now = _now_iso()
        job.status = status
        job.progress_pct = max(0, min(100, progress_pct))
        job.step_message = step_message
        if status == JobStatus.FAILED:
            job.error_message = error_message or step_message
        else:
            job.error_message = None
        job.logs.append(f"[{now}] ({status.value}) {step_message}")
        return save_job(job)


def get_job(job_id: str) -> Optional[JobRecord]:
    with _LOCK:
        if job_id in _JOBS_CACHE:
            return _JOBS_CACHE[job_id]
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                record = _deserialize_job(job_file.read_text(encoding="utf-8"))
                _JOBS_CACHE[job_id] = record
                return record
            except Exception:
                return None
        return None


def list_jobs(limit: int = 25) -> List[JobRecord]:
    with _LOCK:
        for f in JOBS_DIR.glob("*.json"):
            jid = f.stem
            if jid not in _JOBS_CACHE:
                try:
                    _JOBS_CACHE[jid] = _deserialize_job(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
        all_jobs = sorted(_JOBS_CACHE.values(), key=lambda j: j.created_at, reverse=True)
        return all_jobs[:limit]
