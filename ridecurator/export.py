"""Export: copy selected originals + generate an FCPXML for Resolve import (spec §8 Export, §5.4-ish keywords).

Copies ORIGINAL full-res files (never proxies). The FCPXML gives free
DaVinci Resolve keyword tagging via File > Import Timeline, without needing
the paid Studio scripting API.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

FCPXML_FPS = 25  # matches both cameras' typical capture frame rate; adjust if yours differ


def _time(seconds: float, fps: int = FCPXML_FPS) -> str:
    frames = round(seconds * fps)
    return f"{frames}/{fps}s"


def export_selected(
    conn,
    output_dir: str,
    sort_key: str = "interest_score",
    descending: bool = True,
) -> list[dict]:
    """Copy every clip marked selected=True into a numbered output folder.
    Returns the exported clips in the order they were copied (for FCPXML generation)."""
    from ridecurator import db

    clips = [c for c in db.get_all_clips(conn) if c.get("selected")]
    clips.sort(key=lambda c: c.get(sort_key) or 0, reverse=descending)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for i, clip in enumerate(clips, 1):
        src = Path(clip["filepath"])
        dst = output_dir / f"{i:03d}_{clip['camera']}_{src.name}"
        shutil.copy2(src, dst)
        exported.append({**clip, "export_path": str(dst)})

    return exported


def generate_fcpxml(exported_clips: list[dict], fcpxml_path: str, fps: int = FCPXML_FPS) -> str:
    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    fmt_id = "r1"
    ET.SubElement(
        resources, "format", id=fmt_id, name=f"FFVideoFormat1080p{fps}",
        frameDuration=f"1/{fps}s", width="1920", height="1080",
    )

    for i, clip in enumerate(exported_clips, 1):
        duration = _time(clip.get("duration") or 0.0, fps)
        ET.SubElement(
            resources, "asset",
            id=f"a{i}", name=Path(clip["export_path"]).name,
            src=Path(clip["export_path"]).resolve().as_uri(),
            start="0s", duration=duration, hasVideo="1", hasAudio="1", format=fmt_id,
        )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="Uluru Ride Selects")
    project = ET.SubElement(event, "project", name="Uluru Ride Selects Timeline")

    total_duration = sum(c.get("duration") or 0.0 for c in exported_clips)
    sequence = ET.SubElement(
        project, "sequence", format=fmt_id, duration=_time(total_duration, fps),
    )
    spine = ET.SubElement(sequence, "spine")

    offset_seconds = 0.0
    for i, clip in enumerate(exported_clips, 1):
        duration_seconds = clip.get("duration") or 0.0
        asset_clip = ET.SubElement(
            spine, "asset-clip",
            ref=f"a{i}", name=Path(clip["export_path"]).name,
            offset=_time(offset_seconds, fps), duration=_time(duration_seconds, fps),
            start="0s",
        )
        tags = clip.get("tags") or []
        if tags:
            ET.SubElement(
                asset_clip, "keyword",
                start="0s", duration=_time(duration_seconds, fps),
                value=", ".join(tags),
            )
        offset_seconds += duration_seconds

    xml_str = minidom.parseString(ET.tostring(fcpxml)).toprettyxml(indent="  ")
    Path(fcpxml_path).write_text(xml_str, encoding="utf-8")
    return str(fcpxml_path)
