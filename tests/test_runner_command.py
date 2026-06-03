from pathlib import Path

from bernini_runtime.config import JobSpec, RuntimeSettings
from bernini_runtime.runner import _build_command


def test_build_single_gpu_command_uses_upstream_entrypoint(tmp_path: Path):
    job = JobSpec(
        job_id="job-1",
        task_type="v2v",
        guidance_mode="v2v_apg",
        prompt="Add snow.",
        seed=123,
    )
    settings = RuntimeSettings(model_dir=tmp_path / "model")

    command = _build_command(job, settings, tmp_path / "case.json")

    assert command[:2] == ["python", "infer_single_gpu.py"]
    assert command[command.index("--config") + 1] == str(tmp_path / "model")
    assert command[command.index("--seed") + 1] == "123"


def test_build_multi_gpu_command_uses_torchrun_multi_entrypoint(tmp_path: Path):
    job = JobSpec(
        job_id="job-1",
        task_type="v2v",
        guidance_mode="v2v_apg",
        prompt="Add snow.",
        num_gpus=8,
    )
    settings = RuntimeSettings(model_dir=tmp_path / "model")

    command = _build_command(job, settings, tmp_path / "case.json")

    assert command[:3] == ["torchrun", "--nproc-per-node", "8"]
    assert command[3] == "infer_multi_gpu.py"
    assert command[-2:] == ["--ulysses", "8"]
