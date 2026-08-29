"""Near-duplicate clustering (spec §5.3).

The spec names videoduplicatefinder (VDF), but VDF is a Windows GUI app with
no CLI/scripting interface, so it can't be driven from this pipeline. This
module is a from-scratch stand-in that reaches the same outcome — clusters
of near-identical clips, surfaced for confirmation, never auto-discarded —
using perceptual-hash comparison on sampled proxy frames instead of VDF's
ONNX embedding model. If you want a second opinion, you can still point the
real VDF app at the proxy folder by hand.
"""

import cv2
import imagehash
from PIL import Image

from ridecurator.config import DEDUP_HASH_SIZE, DEDUP_MAX_HAMMING_DISTANCE, DEDUP_SAMPLE_FRAMES


def compute_fingerprint(proxy_path: str, sample_frames: int = DEDUP_SAMPLE_FRAMES) -> list[imagehash.ImageHash]:
    """A handful of perceptual hashes sampled across the clip."""
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    import numpy as np
    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)

    hashes = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hashes.append(imagehash.phash(Image.fromarray(rgb), hash_size=DEDUP_HASH_SIZE))
    cap.release()
    return hashes


def fingerprint_distance(fp_a: list[imagehash.ImageHash], fp_b: list[imagehash.ImageHash]) -> float:
    """Average, over fp_a's frames, of the closest match in fp_b. Lower = more similar."""
    if not fp_a or not fp_b:
        return float("inf")
    return sum(min(h_a - h_b for h_b in fp_b) for h_a in fp_a) / len(fp_a)


def cluster_duplicates(
    fingerprints: dict[str, list[imagehash.ImageHash]],
    max_distance: int = DEDUP_MAX_HAMMING_DISTANCE,
) -> dict[str, str]:
    """Union-find clustering. Returns {clip_id: dup_group_id}.

    Clips with no near-duplicate get their own single-member group (dup_group_id
    still set, so the UI can treat "group of one" and "no group" the same way).
    """
    ids = list(fingerprints.keys())
    parent = {cid: cid for cid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            if fingerprint_distance(fingerprints[id_a], fingerprints[id_b]) <= max_distance:
                union(id_a, id_b)

    return {cid: f"grp_{find(cid)}" for cid in ids}


def mark_best_of_group(clips: list[dict]) -> None:
    """Mutates clips in place: sets is_best_of_group=True on the highest
    interest_score clip within each dup_group_id. Call after scoring (spec §5.3:
    "suggested best of cluster, by score, once scoring exists").
    """
    groups: dict[str, list[dict]] = {}
    for c in clips:
        if c.get("dup_group_id"):
            groups.setdefault(c["dup_group_id"], []).append(c)

    for members in groups.values():
        for c in members:
            c["is_best_of_group"] = False
        best = max(members, key=lambda c: c.get("interest_score") or 0.0)
        best["is_best_of_group"] = True
