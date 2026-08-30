"""Near-duplicate clustering (spec §5.3).

The spec names videoduplicatefinder (VDF), but VDF is a Windows GUI app with
no CLI/scripting interface, so it can't be driven from this pipeline. This
module is a from-scratch stand-in that reaches the same outcome — clusters
of near-identical clips, surfaced for confirmation, never auto-discarded —
using two complementary signals instead of VDF's ONNX embedding model:

- perceptual hash on sampled frames (fast, catches near-pixel-identical clips)
- a pretrained CNN's pooled feature per sampled frame, cosine-compared (catches
  "practically the same shot" even across exposure/framing differences that
  perceptual hash misses — reuses torchvision, already a dependency for torch)

Two clips are clustered if EITHER signal says they match. If you want a
second opinion, you can still point the real VDF app at the proxy folder by
hand.
"""

import cv2
import imagehash
import numpy as np
from PIL import Image

from ridecurator.config import (
    DEDUP_EMBEDDING_SAMPLE_FRAMES,
    DEDUP_EMBEDDING_SIMILARITY_THRESHOLD,
    DEDUP_HASH_SIZE,
    DEDUP_MAX_HAMMING_DISTANCE,
    DEDUP_SAMPLE_FRAMES,
)

_embedding_model = None
_embedding_transform = None


def _load_embedding_model(device: str):
    global _embedding_model, _embedding_transform
    if _embedding_model is None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        model.to(device)
        _embedding_model = model
        _embedding_transform = weights.transforms()
    return _embedding_model, _embedding_transform


def compute_embedding(
    proxy_path: str, sample_frames: int = DEDUP_EMBEDDING_SAMPLE_FRAMES, device: str = "cuda"
) -> np.ndarray | None:
    """Mean-pooled, L2-normalized CNN feature across sampled frames — one
    vector per clip. None if the clip couldn't be read at all."""
    import torch

    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None
    indices = np.linspace(0, total_frames - 1, num=min(sample_frames, total_frames), dtype=int)

    model, transform = _load_embedding_model(device)
    features = []
    with torch.no_grad():
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            feat = model.avgpool(model.features(x)).flatten(1)
            features.append(feat.cpu().numpy()[0])
    cap.release()

    if not features:
        return None
    mean_feat = np.mean(features, axis=0)
    norm = np.linalg.norm(mean_feat)
    return mean_feat / norm if norm > 0 else mean_feat


def embedding_similarity(emb_a: np.ndarray | None, emb_b: np.ndarray | None) -> float:
    """Cosine similarity of two L2-normalized embeddings. -1..1, higher = more similar."""
    if emb_a is None or emb_b is None:
        return -1.0
    return float(np.dot(emb_a, emb_b))


def compute_fingerprint(proxy_path: str, sample_frames: int = DEDUP_SAMPLE_FRAMES) -> list[imagehash.ImageHash]:
    """A handful of perceptual hashes sampled across the clip."""
    cap = cv2.VideoCapture(proxy_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

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
    embeddings: dict[str, np.ndarray | None] | None = None,
    max_distance: int = DEDUP_MAX_HAMMING_DISTANCE,
    min_embedding_similarity: float = DEDUP_EMBEDDING_SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Union-find clustering. Returns {clip_id: dup_group_id}.

    Two clips are grouped if EITHER the perceptual-hash distance is within
    max_distance OR the embedding cosine similarity is above
    min_embedding_similarity — the two signals catch different kinds of
    "practically identical" (near-pixel-identical vs same-shot-different-look).

    Clips with no near-duplicate get their own single-member group (dup_group_id
    still set, so the UI can treat "group of one" and "no group" the same way).
    """
    ids = list(fingerprints.keys())
    embeddings = embeddings or {}
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
            phash_match = fingerprint_distance(fingerprints[id_a], fingerprints[id_b]) <= max_distance
            embed_match = (
                embedding_similarity(embeddings.get(id_a), embeddings.get(id_b))
                >= min_embedding_similarity
            )
            if phash_match or embed_match:
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
