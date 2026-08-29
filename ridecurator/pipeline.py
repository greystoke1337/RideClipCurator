"""Orchestration: chains the per-clip analysis passes into the SQLite index.

Each stage is a plain function callable on its own (spec §10.3: "same
functions get called by Streamlit buttons later, avoiding a rewrite"). The
Process tab calls these directly, either one at a time or via run_all().

progress_cb, if given, is called as progress_cb(stage_name, done, total) so
the UI can show a progress bar + inline log per stage (spec §8, Tab 1).
"""

from pathlib import Path
from typing import Callable, Optional

from ridecurator import audio, color, db, dedup, ingest, motion, proxy, scoring, sync, tagging
from ridecurator.tagging import has_motorcycle_tag

ProgressCB = Optional[Callable[[str, int, int], None]]


def _report(cb: ProgressCB, stage: str, done: int, total: int) -> None:
    if cb:
        cb(stage, done, total)


def stage_ingest(conn, raw_dir: str, progress_cb: ProgressCB = None) -> None:
    clips = ingest.scan_folder(raw_dir)
    for i, clip in enumerate(clips, 1):
        db.upsert_clip(conn, clip)
        _report(progress_cb, "ingest", i, len(clips))


def stage_proxy(conn, work_dir: str, progress_cb: ProgressCB = None) -> None:
    proxy_dir = Path(work_dir) / "proxies"
    thumb_dir = Path(work_dir) / "thumbnails"
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        proxy_path = proxy.build_proxy(clip["filepath"], proxy_dir)
        thumb_path = proxy.make_thumbnail(proxy_path, thumb_dir)
        db.upsert_clip(conn, {
            "clip_id": clip["clip_id"],
            "proxy_path": proxy_path,
            "thumbnail_path": thumb_path,
        })
        _report(progress_cb, "proxy", i, len(clips))


def stage_sync(conn, camera_offset_seconds: float = 0.0, progress_cb: ProgressCB = None) -> dict[str, list[str]]:
    clips = db.get_all_clips(conn)
    overlaps = sync.find_overlaps(clips, camera_offset_seconds=camera_offset_seconds)
    _report(progress_cb, "sync", 1, 1)
    return overlaps


def stage_motion(conn, progress_cb: ProgressCB = None) -> None:
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        if not clip.get("proxy_path"):
            continue
        score = motion.steadiness_score(clip["proxy_path"])
        db.upsert_clip(conn, {"clip_id": clip["clip_id"], "steadiness_score": score})
        _report(progress_cb, "motion", i, len(clips))


def stage_color(conn, progress_cb: ProgressCB = None) -> None:
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        if not clip.get("proxy_path"):
            continue
        flag = color.golden_hour_flag(clip["proxy_path"])
        db.upsert_clip(conn, {"clip_id": clip["clip_id"], "golden_hour": flag})
        _report(progress_cb, "color", i, len(clips))


def stage_tagging(conn, ram_checkpoint: str, device: str = "cuda", progress_cb: ProgressCB = None) -> None:
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        if not clip.get("proxy_path"):
            continue
        tags = tagging.tag_clip(clip["proxy_path"], ram_checkpoint, device=device)
        db.upsert_clip(conn, {"clip_id": clip["clip_id"], "tags": tags})
        _report(progress_cb, "tagging", i, len(clips))


def stage_audio(conn, work_dir: str, whisper_model_size: str = "base", progress_cb: ProgressCB = None) -> None:
    wav_dir = Path(work_dir) / "audio"
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        if not clip.get("proxy_path"):
            continue
        wav_path = audio.extract_audio(clip["proxy_path"], wav_dir)
        has_speech = audio.detect_speech(wav_path)
        transcript = None
        if audio.has_decent_audio(wav_path):
            transcript = audio.transcribe(wav_path, model_size=whisper_model_size)
        db.upsert_clip(conn, {
            "clip_id": clip["clip_id"],
            "has_speech": has_speech,
            "transcript": transcript,
        })
        _report(progress_cb, "audio", i, len(clips))


def stage_other_bike(conn, overlaps: dict[str, list[str]], progress_cb: ProgressCB = None) -> None:
    """Compound signal (spec §5.7): sync overlap + RAM's motorcycle tag on this clip."""
    clips = db.get_all_clips(conn)
    for i, clip in enumerate(clips, 1):
        visible = bool(overlaps.get(clip["clip_id"])) and has_motorcycle_tag(clip.get("tags") or [])
        db.upsert_clip(conn, {"clip_id": clip["clip_id"], "other_bike_visible": visible})
        _report(progress_cb, "other_bike", i, len(clips))


def stage_dedup(conn, progress_cb: ProgressCB = None) -> None:
    clips = db.get_all_clips(conn)
    fingerprints = {}
    for i, clip in enumerate(clips, 1):
        if clip.get("proxy_path"):
            fingerprints[clip["clip_id"]] = dedup.compute_fingerprint(clip["proxy_path"])
        _report(progress_cb, "dedup_fingerprint", i, len(clips))

    groups = dedup.cluster_duplicates(fingerprints)
    for i, (clip_id, group_id) in enumerate(groups.items(), 1):
        db.upsert_clip(conn, {"clip_id": clip_id, "dup_group_id": group_id})
        _report(progress_cb, "dedup_cluster", i, len(groups))


def stage_scoring(conn, progress_cb: ProgressCB = None) -> None:
    clips = db.get_all_clips(conn)
    scoring.score_against_full_set(clips)
    dedup.mark_best_of_group(clips)
    for i, clip in enumerate(clips, 1):
        db.upsert_clip(conn, {
            "clip_id": clip["clip_id"],
            "interest_score": clip["interest_score"],
            "score_breakdown": clip["score_breakdown"],
            "is_best_of_group": clip.get("is_best_of_group", False),
        })
        _report(progress_cb, "scoring", i, len(clips))


def run_all(
    conn,
    raw_dir: str,
    work_dir: str,
    ram_checkpoint: str,
    device: str = "cuda",
    camera_offset_seconds: float = 0.0,
    whisper_model_size: str = "base",
    progress_cb: ProgressCB = None,
) -> None:
    """Runs every stage in dependency order. This is the "Run all" button (spec §8, Tab 1)."""
    stage_ingest(conn, raw_dir, progress_cb)
    stage_proxy(conn, work_dir, progress_cb)
    overlaps = stage_sync(conn, camera_offset_seconds, progress_cb)
    stage_motion(conn, progress_cb)
    stage_color(conn, progress_cb)
    stage_tagging(conn, ram_checkpoint, device, progress_cb)
    stage_audio(conn, work_dir, whisper_model_size, progress_cb)
    stage_other_bike(conn, overlaps, progress_cb)
    stage_dedup(conn, progress_cb)
    stage_scoring(conn, progress_cb)
