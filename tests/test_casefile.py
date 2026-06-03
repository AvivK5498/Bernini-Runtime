from pathlib import Path

from bernini_runtime.casefile import build_case_file
from bernini_runtime.config import JobSpec


def test_build_case_file_for_v2v(tmp_path: Path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    video = media_dir / "video_0_input.mp4"
    video.write_bytes(b"video")
    job = JobSpec(
        job_id="job-1",
        task_type="v2v",
        guidance_mode="v2v_apg",
        prompt="Add a snowman.",
    )

    case = build_case_file(
        job,
        media_dir=media_dir,
        output_path=tmp_path / "out.mp4",
        case_path=tmp_path / "case.json",
    )

    assert case["task_type"] == "v2v"
    assert case["video"] == str(video)
    assert case["output"] == str(tmp_path / "out.mp4")
    assert (tmp_path / "case.json").exists()

