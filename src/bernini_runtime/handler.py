from __future__ import annotations

import traceback

import runpod

from .config import PayloadError, RuntimeSettings, validate_payload
from .runner import run_job


def handler(event):
    settings = RuntimeSettings()
    raw_input = event.get("input", {}) if isinstance(event, dict) else {}
    try:
        job = validate_payload(raw_input, job_id=str(event.get("id", "local")))
        return run_job(job, settings)
    except PayloadError as exc:
        return {"ok": False, "error_type": "VALIDATION", "error": str(exc)}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc()[-8000:],
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

