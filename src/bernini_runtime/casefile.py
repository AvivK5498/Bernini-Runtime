from __future__ import annotations

import json
from pathlib import Path

from .config import JobSpec


def build_case_file(job: JobSpec, *, media_dir: Path, output_path: Path, case_path: Path) -> dict:
    case: dict = {
        "task_type": job.task_type,
        "guidance_mode": job.guidance_mode,
        "prompt": job.prompt,
        "output": str(output_path),
    }

    videos = sorted(media_dir.glob("video_*"))
    images = sorted(media_dir.glob("image_*"))

    if videos:
        case["video"] = str(videos[0]) if len(videos) == 1 else [str(path) for path in videos]
    if images:
        case["image" if len(images) == 1 and job.task_type == "i2i" else "images"] = (
            str(images[0]) if len(images) == 1 and job.task_type == "i2i" else [str(path) for path in images]
        )

    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case

