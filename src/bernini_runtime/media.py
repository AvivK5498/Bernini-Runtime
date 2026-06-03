from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


class MediaError(RuntimeError):
    """Raised for media download/upload failures."""


def safe_filename(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        name = fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or fallback


def download_file(url: str, output_path: Path, *, timeout_s: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        if response.status_code >= 400:
            raise MediaError(f"download failed HTTP {response.status_code}: {url}")
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if output_path.stat().st_size == 0:
        raise MediaError(f"download produced an empty file: {url}")
    return output_path


def upload_to_tmpfiles(path: Path, *, api_url: str, timeout_s: int = 300) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, mime)}
        response = requests.post(api_url, files=files, timeout=timeout_s)
    if response.status_code >= 400:
        raise MediaError(f"tmpfiles upload failed HTTP {response.status_code}: {response.text[:400]}")
    body = response.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict) or not data.get("url"):
        raise MediaError(f"tmpfiles response did not include data.url: {body}")
    url = str(data["url"])
    download_url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/", 1)
    return {"url": url, "download_url": download_url}

