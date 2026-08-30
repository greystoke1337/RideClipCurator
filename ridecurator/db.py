"""SQLite index — one row per clip (spec §6)."""

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    filepath TEXT NOT NULL,
    proxy_path TEXT,
    camera TEXT,
    timestamp TEXT,
    duration REAL,
    tags TEXT,                 -- JSON list[str]
    transcript TEXT,
    has_speech INTEGER,
    steadiness_score REAL,
    camera_direction TEXT,      -- "forward" | "backward" | NULL ("unclear")
    flow_coherence REAL,        -- how well optical flow fits a radial expansion/contraction model
    mount_type TEXT,            -- "mounted" | "handheld" | NULL ("unclear")
    other_bike_visible INTEGER,
    dup_group_id TEXT,
    is_best_of_group INTEGER,
    golden_hour INTEGER,
    thumbnail_path TEXT,
    interest_score REAL,
    score_breakdown TEXT,       -- JSON dict[str, float], the "why" behind interest_score
    reviewed INTEGER DEFAULT 0,
    selected INTEGER DEFAULT 0
);
"""

# Columns stored as JSON text in the DB but exposed as Python objects in dicts.
_JSON_COLUMNS = {"tags", "score_breakdown"}
# Columns stored as 0/1 but exposed as bool.
_BOOL_COLUMNS = {
    "has_speech",
    "other_bike_visible",
    "is_best_of_group",
    "golden_hour",
    "reviewed",
    "selected",
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a DB was first created — CREATE TABLE IF
    NOT EXISTS only runs once, so existing DBs need this to pick up new
    fields (e.g. an older pilot.db)."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(clips)")}
    new_columns = {
        "camera_direction": "TEXT",
        "flow_coherence": "REAL",
        "mount_type": "TEXT",
    }
    for name, sql_type in new_columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {name} {sql_type}")
    conn.commit()


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    _migrate(conn)
    return conn


def _encode(clip: dict[str, Any]) -> dict[str, Any]:
    row = dict(clip)
    for col in _JSON_COLUMNS:
        if col in row and row[col] is not None:
            row[col] = json.dumps(row[col])
    for col in _BOOL_COLUMNS:
        if col in row and row[col] is not None:
            row[col] = 1 if row[col] else 0
    return row


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    clip = dict(row)
    for col in _JSON_COLUMNS:
        if clip.get(col):
            clip[col] = json.loads(clip[col])
    for col in _BOOL_COLUMNS:
        if clip.get(col) is not None:
            clip[col] = bool(clip[col])
    return clip


def upsert_clip(conn: sqlite3.Connection, clip: dict[str, Any]) -> None:
    """Insert a clip, or merge new fields into an existing row.

    Analysis passes run independently and each only knows about its own
    fields, so this only overwrites columns actually present in `clip`.
    """
    existing = get_clip(conn, clip["clip_id"])
    merged = {**existing, **clip} if existing else clip
    row = _encode(merged)
    columns = list(row.keys())
    placeholders = ", ".join(f":{c}" for c in columns)
    updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "clip_id")
    conn.execute(
        f"""
        INSERT INTO clips ({", ".join(columns)}) VALUES ({placeholders})
        ON CONFLICT(clip_id) DO UPDATE SET {updates}
        """,
        row,
    )
    conn.commit()


def get_clip(conn: sqlite3.Connection, clip_id: str) -> dict[str, Any] | None:
    cur = conn.execute("SELECT * FROM clips WHERE clip_id = ?", (clip_id,))
    row = cur.fetchone()
    return _decode(row) if row else None


def get_all_clips(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute("SELECT * FROM clips ORDER BY timestamp")
    return [_decode(r) for r in cur.fetchall()]


def set_review(conn: sqlite3.Connection, clip_id: str, *, reviewed: bool | None = None,
               selected: bool | None = None) -> None:
    fields, values = [], []
    if reviewed is not None:
        fields.append("reviewed = ?")
        values.append(1 if reviewed else 0)
    if selected is not None:
        fields.append("selected = ?")
        values.append(1 if selected else 0)
    if not fields:
        return
    values.append(clip_id)
    conn.execute(f"UPDATE clips SET {', '.join(fields)} WHERE clip_id = ?", values)
    conn.commit()
