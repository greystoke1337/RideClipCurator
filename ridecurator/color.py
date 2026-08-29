"""Cheap color analysis — golden-hour bonus, no ML model (spec §5.8, optional/low priority)."""

import cv2
import numpy as np

from ridecurator.config import COLOR_SAMPLE_FRAMES, GOLDEN_HOUR_HUE_DEG, GOLDEN_HOUR_MIN_SATURATION


def golden_hour_flag(proxy_path: str, sample_frames: int = COLOR_SAMPLE_FRAMES) -> bool:
    """True if the clip's average hue falls in the warm gold/orange band
    with enough saturation to mean "warm light," not just a dim gray frame.
    """
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return False

    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)
    hues, sats = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hues.append(hsv[:, :, 0].mean())
        sats.append(hsv[:, :, 1].mean())
    cap.release()

    if not hues:
        return False

    avg_hue = float(np.mean(hues))       # OpenCV hue range is 0-180 (degrees / 2)
    avg_sat = float(np.mean(sats))       # 0-255
    lo, hi = GOLDEN_HOUR_HUE_DEG
    return (lo / 2 <= avg_hue <= hi / 2) and avg_sat >= GOLDEN_HOUR_MIN_SATURATION
