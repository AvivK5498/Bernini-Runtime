# Bernini Runtime

Mutable Runpod serverless runtime for `AvivK5498/Bernini-Docker`.

The Docker image clones this repository on container start and runs:

```bash
python -m bernini_runtime.handler
```

The worker accepts Runpod payloads under `input`, downloads remote media,
builds a Bernini case JSON, invokes upstream ByteDance Bernini-R, and returns
job metadata plus a temporary output URL when upload succeeds.

## Payload

```json
{
  "input": {
    "task_type": "v2v",
    "guidance_mode": "v2v_apg",
    "prompt": "Add warm golden-hour lighting while preserving the subject.",
    "video_url": "https://example.com/input.mp4",
    "num_frames": 33,
    "fps": 16,
    "max_image_size": 848,
    "num_gpus": 1,
    "seed": 42
  }
}
```

For multi-reference tasks, use `image_urls` and/or `video_urls`.

