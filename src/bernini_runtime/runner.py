from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from .casefile import build_case_file
from .config import JobSpec, RuntimeSettings, VIDEO_TASKS
from .media import download_file, safe_filename, upload_to_tmpfiles


class InferenceError(RuntimeError):
    """Raised when Bernini inference fails."""


def run_job(job: JobSpec, settings: RuntimeSettings) -> dict:
    started = time.time()
    job_dir = settings.work_root / job.job_id
    media_dir = job_dir / "media"
    output_dir = job_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_videos = []
    for index, url in enumerate(job.video_urls):
        name = safe_filename(url, f"input_{index}.mp4")
        downloaded_videos.append(download_file(url, media_dir / f"video_{index}_{name}"))

    downloaded_images = []
    for index, url in enumerate(job.image_urls):
        name = safe_filename(url, f"input_{index}.png")
        downloaded_images.append(download_file(url, media_dir / f"image_{index}_{name}"))

    suffix = ".mp4" if job.task_type in VIDEO_TASKS else ".png"
    output_path = output_dir / f"bernini_{job.job_id}{suffix}"
    case_path = job_dir / "case.json"
    case = build_case_file(job, media_dir=media_dir, output_path=output_path, case_path=case_path)

    _ensure_model(settings)
    command = _build_command(job, settings, case_path)
    proc = subprocess.run(
        command,
        cwd=settings.bernini_dir,
        env=_subprocess_env(settings),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=job.timeout_s or settings.default_timeout_s,
    )
    log_path = job_dir / "bernini.log"
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

    if proc.returncode != 0:
        raise InferenceError(f"Bernini exited {proc.returncode}. Log tail:\n{proc.stdout[-4000:]}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise InferenceError(f"Bernini finished but output file is missing or empty: {output_path}")

    upload = None
    try:
        upload = upload_to_tmpfiles(output_path, api_url=settings.tmpfiles_upload_url)
    except Exception as exc:  # Return local path if tmpfiles is unavailable.
        upload = {"error": str(exc)}

    return {
        "ok": True,
        "job_id": job.job_id,
        "task_type": job.task_type,
        "case": case,
        "command": command,
        "output_path": str(output_path),
        "output_size": output_path.stat().st_size,
        "output_upload": upload,
        "downloaded_videos": [str(path) for path in downloaded_videos],
        "downloaded_images": [str(path) for path in downloaded_images],
        "log_path": str(log_path),
        "elapsed_s": round(time.time() - started, 2),
    }


def _ensure_model(settings: RuntimeSettings) -> None:
    if (settings.model_dir / "model_index.json").exists():
        return
    settings.model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="ByteDance/Bernini-R-Diffusers",
        local_dir=str(settings.model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _build_command(job: JobSpec, settings: RuntimeSettings, case_path: Path) -> list[str]:
    num_gpus = job.num_gpus or settings.default_num_gpus
    script = "infer_multi_gpu.py" if num_gpus > 1 else "infer_single_gpu.py"
    script_args = [
        "python",
        script,
        "--config",
        str(settings.model_dir),
        "--case",
        str(case_path),
        "--num_frames",
        str(job.num_frames or settings.default_num_frames),
        "--fps",
        str(job.fps or settings.default_fps),
        "--max_image_size",
        str(job.max_image_size or settings.default_max_image_size),
    ]
    if job.seed is not None:
        script_args.extend(["--seed", str(job.seed)])
    if job.use_prompt_enhancer:
        script_args.append("--use_pe")
    if num_gpus > 1:
        return ["torchrun", "--nproc-per-node", str(num_gpus), *script_args[1:], "--ulysses", str(num_gpus)]
    return script_args


def _subprocess_env(settings: RuntimeSettings) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(settings.model_dir.parent / ".hf-cache"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env
