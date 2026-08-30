# RideClipCurator

Sorts, tags, scores, and curates ~2-3 hours of motorcycle road-trip footage
(shot on a GoPro Hero 10 and a DJI Osmo Action 4, on two different bikes)
down to a curated set of selects ready for editing in DaVinci Resolve.

Full spec: [`docs/spec.md`](docs/spec.md). Everything below is scoped to that
document — read it if you want the "why" behind a design choice.

## This is not a cloud tool

This pipeline needs your GPU and your actual footage files, so it's built to
run **on your own PC**, not in a cloud sandbox. This repo holds the code;
**[`SETUP.md`](SETUP.md)** walks through installing everything and running it
locally, written for Windows and assuming no prior experience with this
particular stack (ffmpeg, CUDA, Streamlit, etc.).

## What's here

```
ridecurator/        Core pipeline — plain, reusable functions, no CLI-only logic.
  ingest.py            scan a folder, pull duration/timestamp/camera via ffprobe
  proxy.py             ffmpeg proxy transcode + thumbnail generation
  sync.py              two-camera timestamp alignment (with drift-offset support)
  motion.py            optical-flow steadiness score and direction of travel
  color.py             golden-hour hue/saturation check
  dedup.py             near-duplicate clustering: perceptual hash + a pretrained
                       CNN embedding (see note below)
  tagging.py           RAM (Recognize Anything Model) content tags
  audio.py             YAMNet speech detection + Whisper transcript
  scoring.py           composite interest_score, with an explainable "why"
  pipeline.py          chains all of the above into the SQLite index
  export.py            copy selected originals + generate FCPXML for Resolve
  db.py                SQLite schema + helpers
  config.py            tunable constants (score weights, thresholds, etc.)
app/
  streamlit_app.py     the actual UI — Process tab + Review tab
scripts/
  run_pilot.py         CLI pilot run against a small clip sample, prints raw output
docs/
  spec.md              the full project spec
data/                  gitignored — raw/, work/ (proxies etc.), output/
models/                gitignored — RAM checkpoint goes here (see SETUP.md)
```

## One deviation from the spec, on purpose

The spec names **videoduplicatefinder** for dedup detection, but it's a
Windows-only GUI app with no CLI or scripting interface — nothing to call
from Python. `dedup.py` implements the same outcome (clusters of
near-identical clips, surfaced for confirmation, never auto-discarded)
natively instead, using perceptual-hash comparison plus a pretrained CNN
embedding (cosine similarity) on sampled proxy frames — the embedding
catches "practically the same shot" even across exposure/framing
differences that perceptual hash alone misses. You can still point the real
app at the proxy folder by hand if you want a second opinion — nothing
here depends on you doing that.

## Status

Scaffolded per the spec's build approach (§10, steps 1 and 3-6): setup docs,
and every core module + the Streamlit UI written as real, working code. What
hasn't happened yet, because it needs your GPU and your real footage:

- **Pilot run (§10.2)** — validate tag/dedup/steadiness quality on ~15-20
  real clips before trusting any of this at scale. Do this first.
- **Score weight tuning (§10.5, §10.8)** — starting weights are in
  `ridecurator/config.py`, expect to revisit them once you see real scores.
- **FCPXML/Resolve import check (§10.6)** — untested against actual Resolve;
  flag it back here if the import doesn't behave as expected.

See [`SETUP.md`](SETUP.md) to get running.
