"""Steadiness scoring via optical flow (spec §5.5).

A primary positive signal, not just a quality filter — smooth footage should
outrank shaky footage of similar content.
"""

import cv2
import numpy as np

from ridecurator.config import MOTION_SAMPLE_FRAMES


def steadiness_score(proxy_path: str, sample_frames: int = MOTION_SAMPLE_FRAMES) -> float:
    """Return a score in [0, 1] — higher means smoother/steadier footage.

    Method: sample frames evenly across the clip, compute dense optical flow
    (Farneback) between consecutive sampled frames, and look at how much the
    flow field's *direction* varies. Camera shake produces flow vectors that
    jump around in direction frame-to-frame; panning/steady motion produces
    flow vectors that stay consistent. High variance -> low steadiness.
    """
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return 0.5  # not enough data to judge; neutral score

    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)

    prev_gray = None
    mean_flows = []  # one average (dx, dy) per frame pair
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mean_flows.append(flow.reshape(-1, 2).mean(axis=0))
        prev_gray = gray
    cap.release()

    if len(mean_flows) < 2:
        return 0.5

    flows = np.array(mean_flows)
    # Jitter = how much the average flow direction/magnitude changes between
    # consecutive sampled pairs. Normalize against overall flow magnitude so
    # a clip that's just panning fast isn't penalized versus a static shot.
    diffs = np.linalg.norm(np.diff(flows, axis=0), axis=1)
    magnitudes = np.linalg.norm(flows, axis=1)
    jitter = diffs.mean() / (magnitudes.mean() + 1e-6)

    # Squash into [0, 1]; jitter of ~2x the mean flow magnitude or more -> ~0.
    score = float(np.clip(1.0 - jitter / 2.0, 0.0, 1.0))
    return score
