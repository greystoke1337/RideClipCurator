"""Motion analysis via optical flow (spec §5.5, plus direction-of-travel and
a mount-type coherence signal derived from the same flow computation).

All three signals come from one shared Farneback optical-flow pass, so a
clip only gets flow-analyzed once regardless of how many signals use it.
"""

import cv2
import numpy as np

from ridecurator.config import DIRECTION_MIN_COHERENCE, MOTION_SAMPLE_FRAMES

# Flow fields are downscaled to this width before computing — plenty of
# resolution for these aggregate signals, and much faster than full 720p.
_ANALYSIS_WIDTH = 160


def _sample_flows(proxy_path: str, sample_frames: int) -> list[np.ndarray]:
    """Dense optical flow between consecutive sampled, downscaled grayscale
    frames. Returns one (h, w, 2) flow field per consecutive pair."""
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)

    prev_gray = None
    flows = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = _ANALYSIS_WIDTH / w
        small = cv2.resize(frame, (_ANALYSIS_WIDTH, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
        prev_gray = gray
    cap.release()
    return flows


def _radial_unit_vectors(h: int, w: int) -> np.ndarray:
    """Unit vector from image center to each pixel, shape (h, w, 2) as (dx, dy)."""
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = xs - cx, ys - cy
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag[mag == 0] = 1.0
    return np.stack([dx / mag, dy / mag], axis=-1)


def analyze_motion(proxy_path: str, sample_frames: int = MOTION_SAMPLE_FRAMES) -> dict:
    """Return steadiness, direction-of-travel, and flow coherence from one
    shared optical-flow pass.

    steadiness_score [0,1]: higher = smoother. Camera shake makes the flow
    field's overall direction/magnitude jump around frame-to-frame;
    panning/steady motion keeps it consistent.

    camera_direction "forward" | "backward" | None ("unclear"): a
    forward-facing camera's flow expands radially outward from the frame
    center as the scene approaches; a rear-facing one contracts inward.
    Classified from the mean dot product between each pixel's flow vector
    and the radial unit vector from center to that pixel.

    flow_coherence [0,1]: how well the flow field actually fits that radial
    model at all. Low coherence (panning, handheld sway, mostly-static
    shots) means camera_direction isn't trustworthy — and is also the
    primary motion signal pipeline.stage_mount_type uses to tell
    bike-mounted footage (coherent, radial despite vibration) from handheld
    (more rotational/erratic).
    """
    flows = _sample_flows(proxy_path, sample_frames)
    if not flows:
        return {"steadiness_score": 0.5, "camera_direction": None, "flow_coherence": 0.0}

    # --- steadiness: jitter in the flow field's overall (mean) direction ---
    mean_flows = np.array([f.reshape(-1, 2).mean(axis=0) for f in flows])
    if len(mean_flows) >= 2:
        diffs = np.linalg.norm(np.diff(mean_flows, axis=0), axis=1)
        magnitudes = np.linalg.norm(mean_flows, axis=1)
        jitter = diffs.mean() / (magnitudes.mean() + 1e-6)
        steadiness = float(np.clip(1.0 - jitter / 2.0, 0.0, 1.0))
    else:
        steadiness = 0.5

    # --- direction + coherence: radial expansion/contraction fit ---
    h, w = flows[0].shape[:2]
    radial = _radial_unit_vectors(h, w)

    dots = np.concatenate([(f * radial).sum(axis=-1).ravel() for f in flows])
    mags = np.concatenate([np.linalg.norm(f, axis=-1).ravel() for f in flows])

    if mags.sum() < 1e-6:
        return {"steadiness_score": steadiness, "camera_direction": None, "flow_coherence": 0.0}

    safe_mags = np.where(mags > 1e-6, mags, 1.0)
    cos_theta = dots / safe_mags  # dot of flow with a *unit* vector = |flow|*cos(theta)

    # Coherence: magnitude-weighted mean of |cos(theta)| — 1.0 means every
    # flow vector points exactly along (or against) the radial direction,
    # 0 means flow is unrelated to it (rotation/panning/noise).
    coherence = float(np.average(np.abs(cos_theta), weights=mags))
    net_alignment = float(np.average(cos_theta, weights=mags))

    if coherence >= DIRECTION_MIN_COHERENCE:
        direction = "forward" if net_alignment > 0 else "backward"
    else:
        direction = None

    return {
        "steadiness_score": steadiness,
        "camera_direction": direction,
        "flow_coherence": coherence,
    }
