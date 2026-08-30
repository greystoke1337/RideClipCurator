"""Uluru Ride Footage Curator — single Streamlit app, two tabs (spec §8).

Run with:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ridecurator import db, export, pipeline, scoring  # noqa: E402

st.set_page_config(page_title="Uluru Ride Footage Curator", layout="wide")

DEFAULT_DB = "data/index.db"
DEFAULT_RAW = "data/raw"
DEFAULT_WORK = "data/work"
DEFAULT_OUTPUT = "data/output"

if "db_path" not in st.session_state:
    st.session_state.db_path = DEFAULT_DB

st.title("Uluru Ride Footage Curator")
tab_process, tab_review = st.tabs(["Process", "Review"])


# ---------------------------------------------------------------- Process ---
with tab_process:
    st.subheader("1. Point at your footage")
    raw_dir = st.text_input("Raw footage folder", value=DEFAULT_RAW)
    work_dir = st.text_input("Working folder (proxies, audio, thumbnails)", value=DEFAULT_WORK)
    db_path = st.text_input("Index database path", value=st.session_state.db_path)
    st.session_state.db_path = db_path

    with st.expander("Model / device settings"):
        ram_checkpoint = st.text_input(
            "RAM checkpoint path", value="models/ram_plus_swin_large_14m.pth",
        )
        device = st.selectbox("Device for RAM / Whisper", ["cuda", "cpu"], index=0)
        whisper_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=1)
        camera_offset = st.number_input(
            "Additional camera clock offset, seconds (residual drift only)",
            value=0.0, step=0.5,
            help=(
                "Per-camera corrections (e.g. GoPro's known Sydney-timezone "
                "offset) are already applied at ingest — see "
                "ridecurator/camera_offsets.json. This is only for extra "
                "drift beyond that; validate on a known-matching moment "
                "first — see sync.py docstring."
            ),
        )

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    st.subheader("2. Run the pipeline")

    STAGES = [
        ("Ingest (scan + metadata)", "ingest"),
        ("Proxy transcode", "proxy"),
        ("Sync (two cameras)", "sync"),
        ("Steadiness (motion)", "motion"),
        ("Golden hour (color)", "color"),
        ("Content tags (RAM)", "tagging"),
        ("Audio (speech + transcript)", "audio"),
        ("Other-bike-visible", "other_bike"),
        ("Mount type", "mount_type"),
        ("Dedup clustering", "dedup"),
        ("Scoring", "scoring"),
    ]

    def make_progress_ui():
        bar = st.progress(0.0)
        log = st.empty()

        def cb(stage, done, total):
            bar.progress(min(done / max(total, 1), 1.0))
            log.text(f"{stage}: {done}/{total}")

        return cb

    cols = st.columns(len(STAGES))
    conn = db.connect(db_path)

    for col, (label, key) in zip(cols, STAGES):
        if col.button(label, key=f"btn_{key}"):
            cb = make_progress_ui()
            try:
                if key == "ingest":
                    pipeline.stage_ingest(conn, raw_dir, cb)
                elif key == "proxy":
                    pipeline.stage_proxy(conn, work_dir, cb)
                elif key == "sync":
                    st.session_state["overlaps"] = pipeline.stage_sync(conn, camera_offset, cb)
                elif key == "motion":
                    pipeline.stage_motion(conn, cb)
                elif key == "color":
                    pipeline.stage_color(conn, cb)
                elif key == "tagging":
                    pipeline.stage_tagging(conn, ram_checkpoint, device, cb)
                elif key == "audio":
                    pipeline.stage_audio(conn, work_dir, whisper_size, cb)
                elif key == "other_bike":
                    overlaps = st.session_state.get("overlaps") or pipeline.stage_sync(conn, camera_offset)
                    pipeline.stage_other_bike(conn, overlaps, cb)
                elif key == "mount_type":
                    pipeline.stage_mount_type(conn, cb)
                elif key == "dedup":
                    pipeline.stage_dedup(conn, device, cb)
                elif key == "scoring":
                    pipeline.stage_scoring(conn, cb)
                st.success(f"{label} done.")
            except Exception as e:
                st.error(f"{label} failed: {e}")

    st.divider()
    if st.button("Run all stages", type="primary"):
        cb = make_progress_ui()
        try:
            pipeline.run_all(
                conn, raw_dir, work_dir, ram_checkpoint, device,
                camera_offset, whisper_size, cb,
            )
            st.success("Full pipeline run complete.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

    clip_count = len(db.get_all_clips(conn))
    st.caption(f"{clip_count} clip(s) currently in the index.")


# ----------------------------------------------------------------- Review ---
with tab_review:
    conn = db.connect(st.session_state.db_path)
    clips = db.get_all_clips(conn)

    if not clips:
        st.info("No clips indexed yet — run the pipeline in the Process tab first.")
    else:
        st.sidebar.header("Filters")
        cameras = sorted({c["camera"] for c in clips})
        camera_filter = st.sidebar.multiselect("Camera", cameras, default=cameras)
        min_score = st.sidebar.slider("Minimum interest score", 0.0, 1.0, 0.0, 0.05)
        all_tags = sorted({t for c in clips for t in (c.get("tags") or [])})
        tag_filter = st.sidebar.multiselect("Tags", all_tags)
        speech_only = st.sidebar.checkbox("Has speech")
        other_bike_only = st.sidebar.checkbox("Other bike visible")
        direction_filter = st.sidebar.multiselect(
            "Camera direction", ["forward", "backward", "unclear"],
            default=["forward", "backward", "unclear"],
        )
        mount_filter = st.sidebar.multiselect(
            "Mount type", ["mounted", "handheld", "unclear"],
            default=["mounted", "handheld", "unclear"],
        )
        sort_by = st.sidebar.selectbox(
            "Sort by", ["interest_score", "steadiness_score", "duration", "timestamp"],
        )

        filtered = [
            c for c in clips
            if c["camera"] in camera_filter
            and (c.get("interest_score") or 0.0) >= min_score
            and (not tag_filter or set(tag_filter) & set(c.get("tags") or []))
            and (not speech_only or c.get("has_speech"))
            and (not other_bike_only or c.get("other_bike_visible"))
            and (c.get("camera_direction") or "unclear") in direction_filter
            and (c.get("mount_type") or "unclear") in mount_filter
        ]
        filtered.sort(key=lambda c: c.get(sort_by) or 0, reverse=True)

        st.write(f"{len(filtered)} clip(s) matching filters")

        # Group by dup_group_id so duplicate clusters render together.
        groups: dict[str, list[dict]] = {}
        for c in filtered:
            groups.setdefault(c.get("dup_group_id") or c["clip_id"], []).append(c)

        selected_ids: set[str] = {c["clip_id"] for c in clips if c.get("selected")}

        for group_id, members in groups.items():
            is_dup_cluster = len(members) > 1
            if is_dup_cluster:
                st.markdown(f"**Duplicate cluster ({len(members)} clips)**")

            for clip in members:
                with st.container(border=True):
                    cols = st.columns([1, 2, 2])
                    with cols[0]:
                        if clip.get("thumbnail_path") and Path(clip["thumbnail_path"]).exists():
                            st.image(clip["thumbnail_path"])
                        if clip.get("proxy_path") and Path(clip["proxy_path"]).exists():
                            with st.expander("Play"):
                                st.video(clip["proxy_path"])

                    with cols[1]:
                        badge = " (best of cluster)" if clip.get("is_best_of_group") else ""
                        st.markdown(f"**{clip['camera']}** — {clip['filepath']}{badge}")
                        st.caption(f"{clip.get('duration', 0):.1f}s · {clip.get('timestamp', '')}")
                        direction = clip.get("camera_direction") or "unclear"
                        mount = clip.get("mount_type") or "unclear"
                        st.caption(f"facing: {direction} · mount: {mount}")
                        st.write(", ".join(f"`{t}`" for t in (clip.get("tags") or [])) or "_no tags_")
                        if clip.get("transcript"):
                            with st.expander("Transcript"):
                                st.write(clip["transcript"])

                    with cols[2]:
                        score = clip.get("interest_score") or 0.0
                        st.metric("Interest score", f"{score:.2f}")
                        breakdown = clip.get("score_breakdown") or {}
                        if breakdown:
                            with st.expander("Why"):
                                for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1]):
                                    st.write(f"{k}: {v:+.2f}")
                        checked = st.checkbox(
                            "Select for export", value=clip["clip_id"] in selected_ids,
                            key=f"select_{clip['clip_id']}",
                        )
                        if checked:
                            selected_ids.add(clip["clip_id"])
                        else:
                            selected_ids.discard(clip["clip_id"])
                        db.set_review(conn, clip["clip_id"], reviewed=True, selected=checked)

        st.divider()
        st.subheader("Export")
        output_dir = st.text_input("Output folder", value=DEFAULT_OUTPUT)
        if st.button("Recompute scores against current selection"):
            scoring.rescore_against_selection(clips, selected_ids)
            for c in clips:
                db.upsert_clip(conn, {
                    "clip_id": c["clip_id"],
                    "interest_score": c["interest_score"],
                    "score_breakdown": c["score_breakdown"],
                })
            st.success("Scores recomputed — landscape-variety bonus now reflects your current selection.")
            st.rerun()

        if st.button("Export selected clips", type="primary"):
            exported = export.export_selected(conn, output_dir)
            if not exported:
                st.warning("No clips selected.")
            else:
                fcpxml_path = str(Path(output_dir) / "uluru_selects.fcpxml")
                export.generate_fcpxml(exported, fcpxml_path)
                st.success(
                    f"Copied {len(exported)} clip(s) to {output_dir} and wrote {fcpxml_path}. "
                    "In Resolve: File > Import Timeline."
                )
