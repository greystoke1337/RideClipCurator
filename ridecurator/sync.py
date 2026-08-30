"""Two-camera sync by file timestamp, not GPS or frame-accurate alignment (spec §5.2).

For every clip, find which clip(s) from the *other* camera overlap in time.
That overlap is what §5.7 (other-bike-visible) and dedup cross-checking key off.

Per-camera clock corrections (e.g. a GoPro left on the wrong timezone) are
applied once at ingest time — see ridecurator/camera_offsets.json — so by the
time clips reach this module their timestamps should already be real UTC.
`camera_offset_seconds` below is for any *additional* residual drift you
measure beyond that (find two clips you know were shot at the same
real-world moment and pass the observed gap).
"""

from datetime import datetime, timedelta
from typing import Any

from ridecurator.config import SYNC_TOLERANCE_SECONDS


def _parse_ts(clip: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(clip["timestamp"])


def find_overlaps(
    clips: list[dict[str, Any]],
    tolerance_seconds: float = SYNC_TOLERANCE_SECONDS,
    camera_offset_seconds: float = 0.0,
) -> dict[str, list[str]]:
    """Return {clip_id: [other_camera_clip_ids that overlap in time]}.

    `camera_offset_seconds` is added to every DJI timestamp before comparing,
    to correct for measured clock drift relative to the GoPro. Validate this
    on a known-matching pair first (see module docstring).
    """
    enriched = []
    for c in clips:
        start = _parse_ts(c)
        if c["camera"] in ("DJI", "drone"):
            start += timedelta(seconds=camera_offset_seconds)
        end = start + timedelta(seconds=c.get("duration", 0.0))
        enriched.append((c["clip_id"], c["camera"], start, end))

    tol = timedelta(seconds=tolerance_seconds)
    overlaps: dict[str, list[str]] = {c["clip_id"]: [] for c in clips}

    for id_a, cam_a, start_a, end_a in enriched:
        for id_b, cam_b, start_b, end_b in enriched:
            if id_a == id_b or cam_a == cam_b:
                continue
            if (start_a - tol) <= end_b and (end_a + tol) >= start_b:
                overlaps[id_a].append(id_b)

    return overlaps


def estimate_offset(known_matching_pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
    """Given one (GoPro clip, DJI clip) pair known to capture the same moment,
    return the seconds to add to DJI timestamps to align it with GoPro.
    """
    gopro_clip, dji_clip = known_matching_pair
    if gopro_clip["camera"] != "GoPro" or dji_clip["camera"] != "DJI":
        raise ValueError("expected (GoPro clip, DJI clip) order")
    return (_parse_ts(gopro_clip) - _parse_ts(dji_clip)).total_seconds()
