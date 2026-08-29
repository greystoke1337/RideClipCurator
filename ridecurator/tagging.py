"""Content tagging via RAM (Recognize Anything Model) (spec §5.4).

This is the primary "what's in this clip" signal. RAM is installed from
source, not pip — see SETUP.md. The checkpoint (ram_plus_swin_large_14m.pth)
is downloaded separately and its path passed in as `checkpoint_path`.

NOTE: this wraps RAM's documented inference API as of the project's spec
date. Verify it works against your installed RAM version during the pilot
phase (spec §10.2) before trusting it at scale — model repo APIs do shift
between versions.
"""

import cv2
import numpy as np
from PIL import Image

from ridecurator.config import TAG_SAMPLE_FRAMES

_model = None
_transform = None


def _load_model(checkpoint_path: str, device: str):
    global _model, _transform
    if _model is None:
        from ram import get_transform
        from ram.models import ram_plus

        _transform = get_transform(image_size=384)
        _model = ram_plus(pretrained=checkpoint_path, image_size=384, vit="swin_l")
        _model.eval()
        _model = _model.to(device)
    return _model, _transform


def _sample_frames(proxy_path: str, sample_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []
    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def tag_clip(
    proxy_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    sample_frames: int = TAG_SAMPLE_FRAMES,
) -> list[str]:
    """Sample frames across the clip, tag each with RAM, union the results."""
    import torch
    from ram import inference_ram as inference

    model, transform = _load_model(checkpoint_path, device)

    all_tags: set[str] = set()
    for frame in _sample_frames(proxy_path, sample_frames):
        image = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            tags_str, _ = inference(image, model)
        all_tags.update(t.strip() for t in tags_str.split("|") if t.strip())

    return sorted(all_tags)


def has_motorcycle_tag(tags: list[str]) -> bool:
    from ridecurator.config import MOTORCYCLE_TAG_WORDS
    return any(t.lower() in MOTORCYCLE_TAG_WORDS for t in tags)
