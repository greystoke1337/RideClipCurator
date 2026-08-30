"""Scan a folder of raw footage and extract per-clip metadata (spec §5, §6).

Uses ffprobe (bundled with ffmpeg) rather than a Python video library so the
only system dependency is the ffmpeg install already required for proxies.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ridecurator.config import VIDEO_EXTENSIONS

# Measured, camera-specific clock corrections (see camera_offsets.json for the
# methodology/evidence behind each value). Applied so every clip's stored
# timestamp is real UTC, regardless of what each camera's clock thought it was —
# downstream sync/scoring can then just compare timestamps directly.
_CAMERA_OFFSETS_PATH = Path(__file__).parent / "camera_offsets.json"
_camera_offsets_cache: dict[str, dict[str, Any]] | None = None


def _load_camera_offsets() -> dict[str, dict[str, Any]]:
    global _camera_offsets_cache
    if _camera_offsets_cache is None:
        _camera_offsets_cache = json.loads(_CAMERA_OFFSETS_PATH.read_text())
    return _camera_offsets_cache


def camera_clock_offset_seconds(camera: str) -> float:
    return _load_camera_offsets().get(camera, {}).get("offset_seconds", 0.0)

# Filename prefixes are the fastest camera signal and don't require probing
# the file. GoPro Hero 10 clips look like GX010123.MP4 / GH010123.MP4.
# DJI Osmo Action (bike-mounted) and a DJI drone both look like DJI_0123.MP4 —
# the filename alone can't tell them apart, so drone falls back to the parent
# folder name (see detect_camera).
_GOPRO_PREFIXES = ("GX", "GH", "GOPR")
_DJI_PREFIXES = ("DJI_",)
_DRONE_FOLDER_HINT = "drone"


def detect_camera(filepath: Path) -> str:
    name = filepath.name.upper()
    if name.startswith(_DJI_PREFIXES):
        if _DRONE_FOLDER_HINT in filepath.parent.name.lower():
            return "drone"
        return "DJI"
    if name.startswith(_GOPRO_PREFIXES):
        return "GoPro"
    return "unknown"


def make_clip_id(filepath: Path) -> str:
    """Stable id derived from the file path, so re-running ingest is idempotent."""
    return hashlib.sha1(str(filepath.resolve()).encode("utf-8")).hexdigest()[:16]


def probe_file(filepath: Path) -> dict[str, Any]:
    """Run ffprobe and pull out duration + capture timestamp."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(filepath),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    fmt = info.get("format", {})

    duration = float(fmt.get("duration", 0.0))

    tags = fmt.get("tags", {})
    raw_ts = tags.get("creation_time")
    if raw_ts:
        # ffprobe reports UTC ISO-8601 with a trailing "Z".
        timestamp = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        from_creation_time = True
    else:
        # Fall back to filesystem mtime minus duration (approximate start time).
        # This is a hint, not a guarantee — see spec §5.2 note on validating sync.
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        timestamp = mtime
        from_creation_time = False

    return {
        "duration": duration,
        "timestamp": timestamp.isoformat(),
        "_timestamp_from_creation_time": from_creation_time,
    }


def scan_folder(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Walk raw_dir, return one metadata dict per video clip found."""
    raw_dir = Path(raw_dir)
    clips = []
    for filepath in sorted(raw_dir.rglob("*")):
        if filepath.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        probed = probe_file(filepath)
        camera = detect_camera(filepath)

        from_creation_time = probed.pop("_timestamp_from_creation_time")
        offset = camera_clock_offset_seconds(camera) if from_creation_time else 0.0
        if offset:
            corrected = datetime.fromisoformat(probed["timestamp"]) + timedelta(seconds=offset)
            probed["timestamp"] = corrected.isoformat()

        clips.append({
            "clip_id": make_clip_id(filepath),
            "filepath": str(filepath),
            "camera": camera,
            **probed,
        })
    return clips
