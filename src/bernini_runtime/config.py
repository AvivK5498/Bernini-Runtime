from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VIDEO_TASKS = {"t2v", "v2v", "mv2v", "rv2v", "r2v", "ads2v"}
IMAGE_TASKS = {"t2i", "i2i"}
SUPPORTED_TASKS = VIDEO_TASKS | IMAGE_TASKS


class PayloadError(ValueError):
    """Raised when a Runpod payload is invalid."""


@dataclass(slots=True)
class RuntimeSettings:
    bernini_dir: Path = Path(os.environ.get("BERNINI_DIR", "/opt/Bernini"))
    model_dir: Path = Path(os.environ.get("BERNINI_MODEL_DIR", "/runpod-volume/models/Bernini-R-Diffusers"))
    work_root: Path = Path(os.environ.get("BERNINI_WORK_ROOT", "/tmp/bernini-jobs"))
    tmpfiles_upload_url: str = os.environ.get("TMPFILES_UPLOAD_URL", "https://tmpfiles.org/api/v1/upload")
    default_num_gpus: int = int(os.environ.get("BERNINI_NUM_GPUS", "1"))
    default_max_image_size: int = int(os.environ.get("BERNINI_MAX_IMAGE_SIZE", "848"))
    default_fps: int = int(os.environ.get("BERNINI_FPS", "16"))
    default_num_frames: int = int(os.environ.get("BERNINI_NUM_FRAMES", "33"))
    default_timeout_s: int = int(os.environ.get("BERNINI_TIMEOUT_S", "7200"))


@dataclass(slots=True)
class JobSpec:
    job_id: str
    task_type: str
    guidance_mode: str
    prompt: str
    video_urls: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    seed: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    max_image_size: int | None = None
    num_gpus: int | None = None
    use_prompt_enhancer: bool = False
    timeout_s: int | None = None


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return value
    raise PayloadError(f"{field_name} must be a URL string or list of URL strings")


def _as_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise PayloadError(f"{field_name} must be greater than 0")
    return parsed


def validate_payload(raw: dict[str, Any], *, job_id: str = "local") -> JobSpec:
    if not isinstance(raw, dict):
        raise PayloadError("input must be an object")

    task_type = str(raw.get("task_type", "")).strip()
    if task_type not in SUPPORTED_TASKS:
        raise PayloadError(f"task_type must be one of {sorted(SUPPORTED_TASKS)}")

    prompt = str(raw.get("prompt", "")).strip()
    if not prompt:
        raise PayloadError("prompt is required")

    guidance_mode = str(raw.get("guidance_mode") or _default_guidance_mode(task_type)).strip()
    video_urls = _as_str_list(raw.get("video_urls", raw.get("video_url")), field_name="video_url(s)")
    image_urls = _as_str_list(raw.get("image_urls", raw.get("image_url")), field_name="image_url(s)")

    if task_type in {"v2v", "mv2v"} and len(video_urls) != 1:
        raise PayloadError(f"{task_type} requires exactly one video_url")
    if task_type == "ads2v" and len(video_urls) < 2:
        raise PayloadError("ads2v requires at least two video_urls")
    if task_type == "rv2v" and (not video_urls or not image_urls):
        raise PayloadError("rv2v requires video_url(s) and image_url(s)")
    if task_type == "r2v" and not image_urls:
        raise PayloadError("r2v requires image_url(s)")
    if task_type == "i2i" and len(image_urls) != 1:
        raise PayloadError("i2i requires exactly one image_url")

    return JobSpec(
        job_id=str(raw.get("job_id") or job_id),
        task_type=task_type,
        guidance_mode=guidance_mode,
        prompt=prompt,
        video_urls=video_urls,
        image_urls=image_urls,
        seed=_as_positive_int(raw.get("seed"), field_name="seed"),
        num_frames=_as_positive_int(raw.get("num_frames"), field_name="num_frames"),
        fps=_as_positive_int(raw.get("fps"), field_name="fps"),
        max_image_size=_as_positive_int(raw.get("max_image_size"), field_name="max_image_size"),
        num_gpus=_as_positive_int(raw.get("num_gpus"), field_name="num_gpus"),
        use_prompt_enhancer=bool(raw.get("use_prompt_enhancer", False)),
        timeout_s=_as_positive_int(raw.get("timeout_s"), field_name="timeout_s"),
    )


def _default_guidance_mode(task_type: str) -> str:
    if task_type in {"t2i", "t2v", "v2v", "mv2v", "ads2v"}:
        return "v2v_apg" if task_type != "t2v" else "t2v_apg"
    if task_type == "r2v":
        return "r2v_apg"
    return "rv2v" if task_type == "rv2v" else "v2v"

