from bernini_runtime.config import PayloadError, validate_payload


def test_validate_v2v_payload_accepts_single_video_url():
    job = validate_payload({
        "task_type": "v2v",
        "prompt": "Keep the scene but make the lighting warmer.",
        "video_url": "https://example.com/input.mp4",
        "num_frames": 33,
    }, job_id="job-test")

    assert job.job_id == "job-test"
    assert job.task_type == "v2v"
    assert job.guidance_mode == "v2v_apg"
    assert job.video_urls == ["https://example.com/input.mp4"]
    assert job.num_frames == 33


def test_validate_rv2v_requires_video_and_image_urls():
    try:
        validate_payload({
            "task_type": "rv2v",
            "prompt": "Replace the shirt with the reference.",
            "video_url": "https://example.com/source.mp4",
        })
    except PayloadError as exc:
        assert "image_url" in str(exc)
    else:
        raise AssertionError("expected PayloadError")

