"""Motion analysis via optical flow (spec §5.5, plus direction-of-travel).

Both signals come from one shared Farneback optical-flow pass, so a clip
only gets flow-analyzed once regardless of how many signals use it.
"""

import cv2
import numpy as np

from ridecurator.config import DIRECTION_MIN_COHERENCE, MOTION_BURST_COUNT, MOTION_BURST_LENGTH

# Flow fields are downscaled to this width before computing — plenty of
# resolution for these aggregate signals, and much faster than full 720p.
_ANALYSIS_WIDTH = 160


def _sample_flow_bursts(
    proxy_path: str, burst_count: int = MOTION_BURST_COUNT, burst_length: int = MOTION_BURST_LENGTH
) -> list[list[np.ndarray]]:
    """Dense optical flow between *consecutive* video frames, in short bursts
    spread across the clip (not one flow field per widely-spaced sample —
    Farneback can't reliably track correspondences across a large gap, e.g.
    every 9th frame at highway speed, which produces wrong-sign flow).

    Returns one list of (h, w, 2) flow fields per burst — kept grouped
    rather than flattened so steadiness's frame-to-frame jitter check (below)
    only compares truly-consecutive pairs, not the artificial jump between
    one burst's last pair and the next burst's first.
    """
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []

    burst_length = min(burst_length, total_frames)
    starts = np.linspace(0, total_frames - burst_length, num=min(burst_count, total_frames - 1), dtype=int)

    bursts = []
    for start in starts:
        burst_flows = []
        prev_gray = None
        for i in range(int(start), int(start) + burst_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
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
                burst_flows.append(flow)
            prev_gray = gray
        if burst_flows:
            bursts.append(burst_flows)
    cap.release()
    return bursts


def _radial_unit_vectors(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Unit vector from (cx, cy) to each pixel, shape (h, w, 2) as (dx, dy)."""
    ys, xs = np.mgrid[0:h, 0:w]
    dx, dy = xs - cx, ys - cy
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag[mag == 0] = 1.0
    return np.stack([dx / mag, dy / mag], axis=-1)


def _evaluate_foe(
    flows_arr: np.ndarray, mag_field: np.ndarray, weight_field: np.ndarray, cx: float, cy: float
) -> tuple[float, float, bool]:
    """How well the flow field radiates from candidate focus-of-expansion
    (cx, cy). Returns (coherence, net_alignment, looks_radial) — see
    analyze_motion's docstring for what each means.

    mag_field is the true per-pixel flow magnitude (used to turn the dot
    product into cos(theta)); weight_field is a winsorized version of it
    used only for averaging, so a handful of extreme-magnitude pixels
    (typically motion-blurred, unreliable Farneback estimates at highway
    speed) can't dominate the average out of proportion to how much of the
    frame they actually are.
    """
    h, w = flows_arr.shape[1:3]
    radial = _radial_unit_vectors(h, w, cx, cy)
    dots = (flows_arr * radial).sum(axis=-1)  # (T, h, w)
    safe_mag = np.where(mag_field > 1e-6, mag_field, 1.0)
    cos_theta = dots / safe_mag

    total_w = weight_field.sum()
    if total_w < 1e-6:
        return 0.0, 0.0, False
    coherence = float((np.abs(cos_theta) * weight_field).sum() / total_w)
    net_alignment = float((cos_theta * weight_field).sum() / total_w)

    mid_x = int(np.clip(round(cx), 1, w - 1))
    mid_y = int(np.clip(round(cy), 1, h - 1))

    def _side_mean(component: np.ndarray, weights: np.ndarray) -> float:
        side_w = weights.sum()
        return float((component * weights).sum() / side_w) if side_w > 1e-6 else 0.0

    x_component, y_component = flows_arr[..., 0], flows_arr[..., 1]
    left_dx = _side_mean(x_component[:, :, :mid_x], weight_field[:, :, :mid_x])
    right_dx = _side_mean(x_component[:, :, mid_x:], weight_field[:, :, mid_x:])
    top_dy = _side_mean(y_component[:, :mid_y, :], weight_field[:, :mid_y, :])
    bottom_dy = _side_mean(y_component[:, mid_y:, :], weight_field[:, mid_y:, :])
    looks_radial = (left_dx * right_dx < 0) or (top_dy * bottom_dy < 0)

    return coherence, net_alignment, looks_radial


def analyze_motion(proxy_path: str) -> dict:
    """Return steadiness, direction-of-travel, and flow coherence from one
    shared optical-flow pass.

    steadiness_score [0,1]: higher = smoother. Camera shake makes the flow
    field's overall direction/magnitude jump around frame-to-frame;
    panning/steady motion keeps it consistent.

    camera_direction "forward" | "backward" | None ("unclear"): a
    forward-facing camera's flow expands radially outward from a focus of
    expansion (FOE) as the scene approaches; a rear-facing one contracts
    inward toward it. The FOE isn't assumed to sit at the image's geometric
    center — a bike frame/pannier filling part of the frame, or a mount
    angled slightly off the direction of travel, shifts it — so a small
    grid of candidate FOE positions is tried and the best-fitting one used.
    A direction is only trusted if, at that best-fit FOE, the flow's
    x-component actually flips sign left vs right of it (or y-component
    top vs bottom) — the real signature of radiating flow. A side-angled
    mount or a curving road can otherwise produce a flow field that's
    mostly one uniform direction (a plain lateral pan), which can look
    deceptively "aligned" by the dot-product alignment test alone without
    actually radiating from anywhere.

    flow_coherence [0,1]: the best-fit FOE's magnitude-weighted alignment
    score. Low coherence (panning, handheld sway, mostly-static shots)
    means camera_direction isn't trustworthy.
    """
    bursts = _sample_flow_bursts(proxy_path)
    flows = [flow for burst in bursts for flow in burst]
    if not flows:
        return {"steadiness_score": 0.5, "camera_direction": None, "flow_coherence": 0.0}

    # --- steadiness: jitter in the flow field's overall (mean) direction ---
    # Diffs computed within each burst only — a jump between one burst's
    # last pair and the next burst's first isn't a real consecutive-frame
    # comparison (see _sample_flow_bursts).
    all_diffs, all_magnitudes = [], []
    for burst in bursts:
        mean_flows = np.array([f.reshape(-1, 2).mean(axis=0) for f in burst])
        all_magnitudes.extend(np.linalg.norm(mean_flows, axis=1))
        if len(mean_flows) >= 2:
            all_diffs.extend(np.linalg.norm(np.diff(mean_flows, axis=0), axis=1))
    if all_diffs:
        jitter = np.mean(all_diffs) / (np.mean(all_magnitudes) + 1e-6)
        steadiness = float(np.clip(1.0 - jitter / 2.0, 0.0, 1.0))
    else:
        steadiness = 0.5

    # --- direction + coherence: best-fit focus-of-expansion (FOE) ---
    flows_arr = np.stack(flows)  # (T, h, w, 2)
    h, w = flows_arr.shape[1:3]
    mag_field = np.linalg.norm(flows_arr, axis=-1)  # (T, h, w)

    if mag_field.sum() < 1e-6:
        return {"steadiness_score": steadiness, "camera_direction": None, "flow_coherence": 0.0}

    # Winsorize: cap each pixel's weight at the 90th-percentile magnitude
    # before averaging. Fast-motion footage (highway speed) motion-blurs
    # roadside detail heavily, and Farneback's dense flow is unreliable on
    # heavily blurred regions — those pixels often have the *largest*
    # magnitude of anywhere in frame, so weighting by raw magnitude lets a
    # small number of corrupted estimates dominate the whole average.
    weight_cap = np.percentile(mag_field, 90)
    weight_field = np.minimum(mag_field, weight_cap) if weight_cap > 1e-6 else mag_field

    # Try a grid of candidate FOE positions rather than assuming image
    # center — see analyze_motion's docstring for why. Restricted to the
    # central region since a genuine forward/rear FOE won't be way out
    # past the frame edge.
    candidates = [
        (fx * w, fy * h)
        for fx in (0.3, 0.4, 0.5, 0.6, 0.7)
        for fy in (0.35, 0.5, 0.65)
    ]
    results = [
        (_evaluate_foe(flows_arr, mag_field, weight_field, cx, cy), (cx, cy))
        for cx, cy in candidates
    ]
    (coherence, net_alignment, looks_radial), _best_center = max(results, key=lambda r: r[0][0])

    if coherence >= DIRECTION_MIN_COHERENCE and looks_radial:
        direction = "forward" if net_alignment > 0 else "backward"
    else:
        direction = None

    return {
        "steadiness_score": steadiness,
        "camera_direction": direction,
        "flow_coherence": coherence,
    }
