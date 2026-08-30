"""Pilot run: process a small folder of real clips and print raw output for
inspection (spec §10.2 — validate signal quality before building UI around it).

Usage:
    python scripts/run_pilot.py --raw data/raw --work data/work --db data/pilot.db \
        --ram-checkpoint models/ram_plus_swin_large_14m.pth
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ridecurator import db, pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--work", default="data/work")
    parser.add_argument("--db", default="data/pilot.db")
    parser.add_argument("--ram-checkpoint", default="models/ram_plus_swin_large_14m.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--camera-offset", type=float, default=0.0)
    args = parser.parse_args()

    def log(stage, done, total):
        print(f"  [{stage}] {done}/{total}", flush=True)

    conn = db.connect(args.db)
    pipeline.run_all(
        conn, args.raw, args.work, args.ram_checkpoint, args.device,
        args.camera_offset, args.whisper_model, log,
    )

    print("\n--- pilot results ---")
    for clip in db.get_all_clips(conn):
        print(json.dumps({
            "camera": clip["camera"],
            "file": Path(clip["filepath"]).name,
            "duration": clip.get("duration"),
            "tags": clip.get("tags"),
            "steadiness": clip.get("steadiness_score"),
            "has_speech": clip.get("has_speech"),
            "other_bike_visible": clip.get("other_bike_visible"),
            "golden_hour": clip.get("golden_hour"),
            "dup_group_id": clip.get("dup_group_id"),
            "interest_score": clip.get("interest_score"),
        }, indent=2))


if __name__ == "__main__":
    main()
