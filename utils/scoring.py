from __future__ import annotations

from typing import Any, Mapping, Sequence
import math


MIN_CONFIDENCE_TEMPERATURE = 0.02
MAX_CONFIDENCE_TEMPERATURE = 0.18


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []

    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)

    if total <= 0:
        return [0.0 for _ in logits]

    return [value / total for value in exps]


def compute_adaptive_temperature(raw_scores: Sequence[float]) -> float:
    scores = [float(score) for score in raw_scores]
    if not scores:
        return 0.10

    spread = max(scores) - min(scores)
    candidate_count = len(scores)

    # 規則：
    # - 分數越扁平、候選越多 -> temperature 越低，拉開差距
    # - 分數本身已經分散 -> temperature 越高，避免過度銳化
    temperature = (
        0.02
        + spread * 0.16
        - min(max(candidate_count - 4, 0) * 0.01, 0.10)
    )

    return float(
        max(MIN_CONFIDENCE_TEMPERATURE, min(MAX_CONFIDENCE_TEMPERATURE, temperature))
    )


def calibrate_scores(
    score_map: Mapping[str, float],
    *,
    temperature: float | None = None,
) -> dict[str, float]:
    if not score_map:
        return {}

    keys = list(score_map.keys())
    raw_scores = [float(score_map[key]) for key in keys]

    resolved_temperature = (
        float(temperature)
        if temperature is not None
        else compute_adaptive_temperature(raw_scores)
    )

    logits = [score / resolved_temperature for score in raw_scores]
    probabilities = _softmax(logits)

    return {
        key: float(probability)
        for key, probability in zip(keys, probabilities)
    }


def build_candidates(
    score_map: Mapping[str, float],
    label_map: Mapping[str, str],
    *,
    temperature: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    normalized_map = calibrate_scores(score_map, temperature=temperature)

    candidates = [
        {
            "value": key,
            "label": label_map[key],
            "score": float(normalized_map.get(key, 0.0)),
        }
        for key in label_map.keys()
    ]

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    return candidates, normalized_map


def pick_multi_selected(
    candidates: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    max_selected: int,
) -> list[str]:
    selected = [
        str(item["value"])
        for item in candidates
        if float(item["score"]) >= threshold
    ][:max_selected]

    if selected:
        return selected

    return [str(candidates[0]["value"])] if candidates else []