"""Proxy transcode: 8-bit, downscaled copies of the 10-bit originals (spec §5.1).

Every analysis pass runs against these proxies. Originals are never touched.
"""

import subprocess
from pathlib import Path

from ridecurator.config import PROXY_CODEC, PROXY_CRF, PROXY_HEIGHT, PROXY_PRESET


def build_proxy(src_path: str | Path, proxy_dir: str | Path) -> str:
    """Transcode src_path into proxy_dir, return the proxy's path."""
    src_path = Path(src_path)
    proxy_dir = Path(proxy_dir)
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = proxy_dir / f"{src_path.stem}.mp4"

    if proxy_path.exists():
        return str(proxy_path)

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src_path),
            "-vf", f"scale=-2:{PROXY_HEIGHT}",
            "-pix_fmt", "yuv420p",           # forces 8-bit output
            "-c:v", PROXY_CODEC,
            "-preset", PROXY_PRESET,
            "-crf", str(PROXY_CRF),
            "-c:a", "aac", "-b:a", "128k",
            str(proxy_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return str(proxy_path)


def make_thumbnail(proxy_path: str | Path, thumb_dir: str | Path) -> str:
    """Grab a single mid-clip frame as a JPEG thumbnail for the review UI."""
    proxy_path = Path(proxy_path)
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"{proxy_path.stem}_thumb.jpg"

    if thumb_path.exists():
        return str(thumb_path)

    duration = _probe_duration(proxy_path)
    # Back off half a second from the end so the seek always lands on a
    # decodable frame — plain duration/2 can overshoot the last frame on
    # very short clips (a fraction-of-a-second clip has no frame at its
    # exact midpoint).
    midpoint = max(0.0, min(duration / 2, duration - 0.5))

    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(midpoint), "-i", str(proxy_path),
            "-frames:v", "1", "-pix_fmt", "yuvj420p", "-q:v", "3", str(thumb_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return str(thumb_path)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip() or 0.0)
