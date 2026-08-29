# Uluru Ride Footage Pipeline — Project Spec

## 1. Purpose

Sort, tag, score, and curate ~2-3 hours of motorcycle road-trip footage (Uluru trip) shot on two cameras across two motorcycles, down to a curated set of selects ready to hand to DaVinci Resolve (free version) for editing.

The footage is mostly short clips (10-20s). Manually scrubbing through all of it to find the interesting moments is the problem this solves.

## 2. Source material

- **Camera A:** GoPro Hero 10 — 10-bit HEVC, has GPS/GPMF telemetry
- **Camera B:** DJI Osmo Action 4 — 10-bit HEVC, has good onboard mic audio, weaker/no GPS telemetry
- Both cameras were recording on the **same ride, on two different motorcycles** — footage needs to be time-synced (by file timestamp, not frame-accurate) so we can detect when the *other* rider/bike appears in one camera's shot.
- Total volume: ~2-3 hours, many short clips.

## 3. What "interesting" means for this trip

Explicitly **not** speed/action-based. Priority signals, in the rider's own words:

1. **Steadiness** — smooth footage over shaky
2. **Landscape variety** — don't want 20 near-identical "empty outback road" clips; variety across the whole selection matters more than any single clip's score
3. **The other motorcycle/rider visible in frame** — e.g. two clips of the same empty outback road: the one where the other bike is visible is more interesting than the one where it's just empty road
4. **People talking** — presence of conversation/speech is a positive signal

No GPS/speed-based interest scoring. No low-light/night special-casing needed (not expected on this trip, revisit if it comes up).

## 4. Processing granularity

- **Whole clip**, not sub-scene. No need to split clips into sub-shots.
- Analysis and review happen on **proxies** (downscaled 8-bit). Original 10-bit files are never modified — final output is a **copy** of selected originals into a new, ordered folder.

## 5. Pipeline stages

```
Raw footage (GoPro + DJI, 10-bit)
   → Proxy transcode (ffmpeg → 8-bit, low-res)
   → Two-camera sync (align by file timestamp)
   → [parallel analysis passes, see below]
   → Merge into SQLite index
   → Streamlit app: review, filter, confirm dedup groups, select
   → Copy selected ORIGINAL (full-res) files into ordered output folder
   → Import into free DaVinci Resolve (folder structure + FCPXML keywords)
```

### 5.1 Proxy transcode
- Tool: **ffmpeg**
- 8-bit, downscaled (e.g. 480-720p), fast-decode codec, for use by every downstream analysis tool

### 5.2 Two-camera sync
- Align clips from both cameras by embedded file timestamp (camera clock), not GPS.
- Goal: for any given clip, know what (if anything) the *other* camera was recording at the same moment. Needed to detect "other bike visible" and to support cross-camera dedup.
- Note: expect clock drift between the two cameras — validate sync accuracy on a couple of known-matching moments before trusting it at scale.

### 5.3 Dedup / near-duplicate detection
- Tool: **videoduplicatefinder** (free, open-source, has AI-matching mode via local ONNX embedding model)
- Run in AI-matching mode against the proxy set.
- Output: clusters of near-identical clips + a suggested "best of cluster" (by score, once scoring exists).
- Do not auto-discard — surface clusters in the review UI for confirmation.

### 5.4 Content tagging
- Tool: **RAM (Recognize Anything Model)** — free, open-vocabulary image tagging, run locally on GPU (3070)
- Run on sampled frames per clip → aggregate into a tag list per clip (e.g. "motorcycle," "dirt road," "rock formation," "kangaroo," "person," "helmet," "dust").
- This is the primary "what's in this clip" signal — replaces the need for a hand-picked detector list (no separate YOLO/CLIP-zero-shot pass needed).
- Tags feed both the review UI (as a filterable transcript) and the landscape-variety scoring.

### 5.5 Motion / steadiness analysis
- Tool: **OpenCV** optical flow
- Output: a steadiness score per clip (low jitter = high score). This is a primary positive signal, not just a quality filter.

### 5.6 Audio analysis
- **YAMNet** (free, local) — audio event classification → flags speech/talking/laughter presence per clip. Primary signal for "people talking."
- **Whisper** (free, local) — full transcript, run only on clips that have decent mic audio (DJI clips primarily; skip GoPro clips with only wind/engine noise — use a quick silence/audio-level check first to avoid wasted compute).

### 5.7 Other-bike-visible detection
- Uses the two-camera sync (5.2) plus RAM's "motorcycle" tag (5.4) to flag clips where the second bike is identifiable in frame.
- This is a compound signal, not a separate model.

### 5.8 (Optional/cheap) Color analysis
- Tool: **OpenCV** hue/color-temperature stats — no ML model needed.
- Flags golden-hour/sunset-sunrise footage as a minor bonus signal. Low priority — include if time allows.

## 6. Data model

One **SQLite** index, one row per clip:

| Field | Description |
|---|---|
| `clip_id` | unique id |
| `filepath` | path to original full-res file |
| `proxy_path` | path to proxy |
| `camera` | GoPro / DJI |
| `timestamp` | capture time (for sync) |
| `duration` | seconds |
| `tags` | list from RAM |
| `transcript` | Whisper output (nullable) |
| `has_speech` | bool, from YAMNet |
| `steadiness_score` | float |
| `other_bike_visible` | bool |
| `dup_group_id` | nullable, from dedup pass |
| `is_best_of_group` | bool |
| `golden_hour` | bool (optional) |
| `thumbnail_path` | for UI display |
| `interest_score` | composite score (see §7) |
| `reviewed` / `selected` | user decision, set via UI |

## 7. Scoring

Composite `interest_score` combines (weights to be tuned during pilot, not fixed upfront):
- Steadiness score (positive weight)
- Other-bike-visible (positive, likely high weight per user's stated preference)
- Has-speech (positive)
- Landscape-variety bonus — **this one is relative to the whole selection, not a per-clip static score**: a clip's tag combination gets a bonus if under-represented among current top-ranked/selected clips. Needs to be computed against the running selection, not in isolation.

Every score should be explainable — store which sub-signals contributed, so the UI can show a short "why" next to the number, not just a bare figure.

## 8. Review UI (Streamlit)

Single app, two tabs — this avoids ever needing a terminal after initial setup.

### Tab 1 — Process
- Point at a folder of raw clips
- Run pipeline stages (either individually or one "Run all" button)
- Progress bar + inline log per stage
- Synchronous execution is fine at this footage volume (few hundred short clips on a 3070) — no need for background job infrastructure

### Tab 2 — Review
- Grid/list of clips: thumbnail, camera, duration, interest score (+ why), tags as chips, transcript snippet (expandable), inline video playback
- Sidebar filters: camera, min score, tags, has-speech, has-other-bike
- Sort by: interest score, steadiness, duration, timestamp
- Dedup clusters shown as groups with a suggested "best of cluster," rest marked as duplicates — user confirms/overrides, never auto-discarded
- Bulk select checkboxes → "export selected" action

### Export
- Copies selected clips' **original full-res files** (not proxies) into a new, ordered/numbered output folder
- Also generates an **FCPXML** file with keyword/marker metadata, importable into free DaVinci Resolve (`File > Import Timeline`) — gives keyword tagging in Resolve without needing the paid Studio scripting API

## 9. Tooling summary (all free, local, GPU-accelerated where relevant on the 3070)

| Purpose | Tool |
|---|---|
| Transcode | ffmpeg |
| Dedup | videoduplicatefinder (AI-matching mode) |
| Content tags | RAM |
| Steadiness | OpenCV (optical flow) |
| Audio events | YAMNet |
| Transcript | Whisper |
| Color/golden hour | OpenCV |
| Index | SQLite |
| UI | Streamlit |
| Export format | FCPXML (+ plain folder copy) |

No paid tools required anywhere in this pipeline. (DaVinci Resolve Studio's scripting API was considered but is not used — free Resolve's native FCPXML import covers the need.)

## 10. Build approach

**Principle: validate signal quality before building UI around it.** The risk in this project is whether the tags/scores are actually useful, not whether Streamlit works.

1. **Setup** — install all tools, confirm GPU (CUDA) is actually used by each model
2. **Pilot** — run each tool standalone against ~15-20 real clips spanning both cameras and a range of content; inspect raw output quality (tags, dedup clusters, steadiness numbers) before writing any integration code
3. **Core modules** — write ingest, sync, and each analysis pass as **plain, reusable functions** (not CLI scripts) — same functions get called by Streamlit buttons later, avoiding a rewrite
4. **Index** — single script/module that runs all analysis functions per clip and writes to SQLite; run against the pilot batch, sanity-check
5. **Streamlit app** — build both tabs against real pilot data (not dummy data); this is where scoring weights will likely need revisiting
6. **Export** — folder-copy + FCPXML generation; test-import into free Resolve to confirm compatibility
7. **Full run** — process all ~2-3 hours
8. **Iterate** — revisit scoring weights and tags based on what the full run surfaces

## 11. Explicit non-goals (for this version)

- No speed/GPS-based scoring
- No low-light/night-specific handling
- No sub-scene splitting (whole-clip granularity only)
- No cloud processing — everything local/free
- Not built as a general-purpose tool — scoped to this trip's specifics (2 bikes, outback landscape, DJI+GoPro pairing); can generalize later if there are future trips like this
