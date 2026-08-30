"""Composite interest_score (spec §7).

Every score is explainable: score_breakdown stores each sub-signal's
contribution so the UI can show a "why" next to the number.

The landscape-variety term is relative, not a static per-clip value — it's
scored against a reference set of clips (either "everything" for the first
pass, or "what's currently selected" as the user builds their selection in
the Review tab), and needs recomputing as that reference set changes.
"""

from typing import Any

from ridecurator.config import SCORE_WEIGHTS


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def landscape_variety_bonus(clip_tags: set[str], reference_tag_sets: list[set[str]]) -> float:
    """1.0 = tag combo unlike anything in the reference set (fresh landscape).
    0.0 = tag combo is a near-exact match of something already there."""
    if not reference_tag_sets:
        return 1.0
    most_similar = max(_jaccard(clip_tags, ref) for ref in reference_tag_sets)
    return 1.0 - most_similar


def compute_interest_score(
    clip: dict[str, Any],
    reference_clips: list[dict[str, Any]],
    weights: dict[str, float] = SCORE_WEIGHTS,
) -> tuple[float, dict[str, float]]:
    """reference_clips is the set the variety bonus is judged against — pass
    every other clip for a first-pass ranking, or just the currently-selected
    clips to score how much a candidate would add to the running selection."""
    clip_tags = set(clip.get("tags") or [])
    reference_tag_sets = [
        set(c.get("tags") or [])
        for c in reference_clips
        if c["clip_id"] != clip["clip_id"]
    ]

    signals = {
        "steadiness": clip.get("steadiness_score") or 0.0,
        "other_bike_visible": 1.0 if clip.get("other_bike_visible") else 0.0,
        "has_speech": 1.0 if clip.get("has_speech") else 0.0,
        "landscape_variety": landscape_variety_bonus(clip_tags, reference_tag_sets),
        "golden_hour": 1.0 if clip.get("golden_hour") else 0.0,
    }

    breakdown = {name: weights.get(name, 0.0) * value for name, value in signals.items()}
    score = sum(breakdown.values())
    return score, breakdown


def score_against_full_set(
    clips: list[dict[str, Any]], weights: dict[str, float] = SCORE_WEIGHTS
) -> None:
    """First-pass scoring: variety judged against the whole dataset. Mutates clips in place."""
    for clip in clips:
        score, breakdown = compute_interest_score(clip, clips, weights)
        clip["interest_score"] = score
        clip["score_breakdown"] = breakdown


def rescore_against_selection(
    clips: list[dict[str, Any]],
    selected_clip_ids: set[str],
    weights: dict[str, float] = SCORE_WEIGHTS,
) -> None:
    """Re-rank un-selected clips by how much they'd add to the current
    selection — call this from the Review tab whenever the user checks/unchecks
    a clip, so variety bonuses reflect what's already picked. Mutates clips in place."""
    selected = [c for c in clips if c["clip_id"] in selected_clip_ids]
    for clip in clips:
        score, breakdown = compute_interest_score(clip, selected, weights)
        clip["interest_score"] = score
        clip["score_breakdown"] = breakdown
