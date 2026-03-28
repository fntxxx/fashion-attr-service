from __future__ import annotations

from typing import Any

from fashion_attr_service.utils.scoring import build_candidates


OCCASION_OPTIONS = [
    ("social", "社交聚會"),
    ("campus_casual", "校園休閒"),
    ("business_casual", "商務休閒"),
    ("professional", "專業職場"),
]

SEASON_OPTIONS = [
    ("spring", "春季"),
    ("summer", "夏季"),
    ("autumn", "秋季"),
    ("winter", "冬季"),
]

SEASON_ALLOWED_DOUBLE_COMBINATIONS = {
    frozenset({"spring", "autumn"}),
}

OCCASION_ALLOWED_DOUBLE_COMBINATIONS = {
    frozenset({"campus_casual", "social"}),
    frozenset({"business_casual", "social"}),
    frozenset({"business_casual", "professional"}),
}

STRICT_OCCASION_KEYS = {"business_casual", "professional"}


def _build_single_or_conditional_double_selection(
    candidates: list[dict[str, Any]],
    *,
    second_score_min: float,
    max_gap: float,
    allowed_pairs: set[frozenset[str]] | None = None,
    strict_keys: set[str] | None = None,
    strict_second_score_min: float | None = None,
    strict_max_gap: float | None = None,
    force_single_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {
            "selected": [],
            "decision": "empty",
            "top1": None,
            "top2": None,
            "top1_score": 0.0,
            "top2_score": 0.0,
            "gap": 0.0,
            "allowed_double": False,
        }

    top1 = candidates[0]
    selected = [str(top1["value"])]

    if len(candidates) < 2:
        return {
            "selected": selected,
            "decision": "single_only_one_candidate",
            "top1": str(top1["value"]),
            "top2": None,
            "top1_score": float(top1["score"]),
            "top2_score": 0.0,
            "gap": 1.0,
            "allowed_double": False,
        }

    top2 = candidates[1]
    top1_value = str(top1["value"])
    top2_value = str(top2["value"])
    top1_score = float(top1["score"])
    top2_score = float(top2["score"])
    gap = top1_score - top2_score
    pair = frozenset({top1_value, top2_value})

    if force_single_keys and top1_value in force_single_keys:
        return {
            "selected": selected,
            "decision": "single_force_top1",
            "top1": top1_value,
            "top2": top2_value,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "gap": gap,
            "allowed_double": False,
        }

    resolved_second_score_min = second_score_min
    resolved_max_gap = max_gap
    if strict_keys and ({top1_value, top2_value} & strict_keys):
        resolved_second_score_min = (
            float(strict_second_score_min)
            if strict_second_score_min is not None
            else second_score_min
        )
        resolved_max_gap = (
            float(strict_max_gap)
            if strict_max_gap is not None
            else max_gap
        )

    pair_allowed = True if allowed_pairs is None else pair in allowed_pairs
    allow_double = (
        pair_allowed
        and top2_score >= resolved_second_score_min
        and gap <= resolved_max_gap
    )

    if allow_double:
        selected.append(top2_value)

    return {
        "selected": selected,
        "decision": "double" if allow_double else "single_top1",
        "top1": top1_value,
        "top2": top2_value,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "gap": gap,
        "allowed_double": allow_double,
        "resolved_second_score_min": resolved_second_score_min,
        "resolved_max_gap": resolved_max_gap,
        "pair_allowed": pair_allowed,
    }


def infer_occasions(main_category: str, fine_category: str) -> dict[str, Any]:
    label_map = {value: label for value, label in OCCASION_OPTIONS}

    base_scores: dict[str, float] = {
        "social": 0.28,
        "campus_casual": 0.32,
        "business_casual": 0.26,
        "professional": 0.20,
    }

    fine_overrides: dict[str, dict[str, float]] = {
        "t_shirt": {
            "campus_casual": 0.88,
            "social": 0.38,
            "business_casual": 0.18,
            "professional": 0.08,
        },
        "shirt": {
            "campus_casual": 0.52,
            "social": 0.58,
            "business_casual": 0.83,
            "professional": 0.72,
        },
        "tank_top": {
            "campus_casual": 0.70,
            "social": 0.48,
            "business_casual": 0.12,
            "professional": 0.05,
        },
        "hoodie": {
            "campus_casual": 0.90,
            "social": 0.26,
            "business_casual": 0.10,
            "professional": 0.05,
        },
        "sweatshirt": {
            "campus_casual": 0.86,
            "social": 0.30,
            "business_casual": 0.15,
            "professional": 0.06,
        },
        "knit_sweater": {
            "campus_casual": 0.62,
            "social": 0.54,
            "business_casual": 0.56,
            "professional": 0.36,
        },
        "cardigan": {
            "campus_casual": 0.42,
            "social": 0.46,
            "business_casual": 0.72,
            "professional": 0.50,
        },
        "denim_jacket": {
            "campus_casual": 0.82,
            "social": 0.42,
            "business_casual": 0.18,
            "professional": 0.08,
        },
        "blazer": {
            "campus_casual": 0.08,
            "social": 0.44,
            "business_casual": 0.88,
            "professional": 0.92,
        },
        "coat": {
            "campus_casual": 0.42,
            "social": 0.50,
            "business_casual": 0.68,
            "professional": 0.64,
        },
        "puffer_jacket": {
            "campus_casual": 0.74,
            "social": 0.24,
            "business_casual": 0.12,
            "professional": 0.08,
        },
        "vest": {
            "campus_casual": 0.44,
            "social": 0.34,
            "business_casual": 0.42,
            "professional": 0.26,
        },
        "windbreaker": {
            "campus_casual": 0.72,
            "social": 0.20,
            "business_casual": 0.10,
            "professional": 0.05,
        },
        "jeans": {
            "campus_casual": 0.88,
            "social": 0.42,
            "business_casual": 0.22,
            "professional": 0.08,
        },
        "trousers": {
            "campus_casual": 0.22,
            "social": 0.40,
            "business_casual": 0.84,
            "professional": 0.92,
        },
        "wide_leg_pants": {
            "campus_casual": 0.28,
            "social": 0.38,
            "business_casual": 0.72,
            "professional": 0.64,
        },
        "leggings": {
            "campus_casual": 0.72,
            "social": 0.18,
            "business_casual": 0.06,
            "professional": 0.04,
        },
        "shorts": {
            "campus_casual": 0.82,
            "social": 0.32,
            "business_casual": 0.08,
            "professional": 0.04,
        },
        "mini_skirt": {
            "campus_casual": 0.58,
            "social": 0.70,
            "business_casual": 0.22,
            "professional": 0.10,
        },
        "midi_skirt": {
            "campus_casual": 0.42,
            "social": 0.66,
            "business_casual": 0.56,
            "professional": 0.28,
        },
        "mini_dress": {
            "campus_casual": 0.36,
            "social": 0.84,
            "business_casual": 0.24,
            "professional": 0.10,
        },
        "midi_dress": {
            "campus_casual": 0.28,
            "social": 0.78,
            "business_casual": 0.48,
            "professional": 0.18,
        },
        "sneakers": {
            "campus_casual": 0.90,
            "social": 0.26,
            "business_casual": 0.12,
            "professional": 0.04,
        },
        "boots": {
            "campus_casual": 0.34,
            "social": 0.44,
            "business_casual": 0.20,
            "professional": 0.08,
        },
        "sandals": {
            "campus_casual": 0.58,
            "social": 0.52,
            "business_casual": 0.08,
            "professional": 0.04,
        },
        "heels": {
            "campus_casual": 0.12,
            "social": 0.76,
            "business_casual": 0.42,
            "professional": 0.22,
        },
        "flats": {
            "campus_casual": 0.26,
            "social": 0.50,
            "business_casual": 0.46,
            "professional": 0.18,
        },
        "bucket_hat": {
            "campus_casual": 0.72,
            "social": 0.24,
            "business_casual": 0.06,
            "professional": 0.02,
        },
        "beanie": {
            "campus_casual": 0.66,
            "social": 0.16,
            "business_casual": 0.04,
            "professional": 0.02,
        },
        "hat": {
            "campus_casual": 0.52,
            "social": 0.30,
            "business_casual": 0.08,
            "professional": 0.04,
        },
    }

    score_map = fine_overrides.get(fine_category, base_scores)
    score_map = dict(score_map)

    outer_like_fine_categories = {
        "denim_jacket",
        "cardigan",
        "blazer",
        "coat",
        "puffer_jacket",
        "vest",
        "windbreaker",
        "sweatshirt",
    }

    if fine_category in outer_like_fine_categories:
        score_map["campus_casual"] = min(score_map.get("campus_casual", 0.0), 0.60)
        score_map["business_casual"] = max(score_map.get("business_casual", 0.0), 0.62)
        score_map["professional"] = max(score_map.get("professional", 0.0), 0.36)

    candidates, normalized_map = build_candidates(score_map, label_map)
    selection = _build_single_or_conditional_double_selection(
        candidates,
        second_score_min=0.22,
        max_gap=0.12,
        allowed_pairs=OCCASION_ALLOWED_DOUBLE_COMBINATIONS,
        strict_keys=STRICT_OCCASION_KEYS,
        strict_second_score_min=0.30,
        strict_max_gap=0.08,
    )
    selected = selection["selected"]

    legacy_style = "casual"
    if "professional" in selected:
        legacy_style = "formal"
    elif fine_category in {"hoodie", "leggings", "windbreaker", "sneakers"}:
        legacy_style = "sport"
    elif "business_casual" in selected:
        legacy_style = "smart_casual"

    return {
        "selected": selected,
        "candidates": candidates,
        "candidateScoreMap": normalized_map,
        "selectionDebug": selection,
        "legacy_style": legacy_style,
    }


def infer_seasons(main_category: str, fine_category: str) -> dict[str, Any]:
    label_map = {value: label for value, label in SEASON_OPTIONS}

    base_scores: dict[str, float] = {
        "spring": 0.55,
        "summer": 0.20,
        "autumn": 0.58,
        "winter": 0.30,
    }

    fine_overrides: dict[str, dict[str, float]] = {
        "tank_top": {
            "spring": 0.36,
            "summer": 0.92,
            "autumn": 0.14,
            "winter": 0.04,
        },
        "t_shirt": {
            "spring": 0.70,
            "summer": 0.76,
            "autumn": 0.48,
            "winter": 0.10,
        },
        "shirt": {
            "spring": 0.72,
            "summer": 0.40,
            "autumn": 0.66,
            "winter": 0.18,
        },
        "hoodie": {
            "spring": 0.44,
            "summer": 0.06,
            "autumn": 0.82,
            "winter": 0.70,
        },
        "sweatshirt": {
            "spring": 0.42,
            "summer": 0.04,
            "autumn": 0.84,
            "winter": 0.72,
        },
        "knit_sweater": {
            "spring": 0.22,
            "summer": 0.02,
            "autumn": 0.74,
            "winter": 0.92,
        },
        "cardigan": {
            "spring": 0.80,
            "summer": 0.08,
            "autumn": 0.62,
            "winter": 0.58,
        },
        "denim_jacket": {
            "spring": 0.54,
            "summer": 0.06,
            "autumn": 0.80,
            "winter": 0.56,
        },
        "blazer": {
            "spring": 0.58,
            "summer": 0.18,
            "autumn": 0.72,
            "winter": 0.44,
        },
        "coat": {
            "spring": 0.10,
            "summer": 0.01,
            "autumn": 0.48,
            "winter": 0.96,
        },
        "puffer_jacket": {
            "spring": 0.04,
            "summer": 0.01,
            "autumn": 0.28,
            "winter": 0.99,
        },
        "vest": {
            "spring": 0.42,
            "summer": 0.26,
            "autumn": 0.66,
            "winter": 0.34,
        },
        "windbreaker": {
            "spring": 0.62,
            "summer": 0.16,
            "autumn": 0.74,
            "winter": 0.32,
        },
        "jeans": {
            "spring": 0.64,
            "summer": 0.18,
            "autumn": 0.74,
            "winter": 0.44,
        },
        "trousers": {
            "spring": 0.62,
            "summer": 0.22,
            "autumn": 0.74,
            "winter": 0.40,
        },
        "wide_leg_pants": {
            "spring": 0.58,
            "summer": 0.30,
            "autumn": 0.70,
            "winter": 0.34,
        },
        "leggings": {
            "spring": 0.26,
            "summer": 0.02,
            "autumn": 0.70,
            "winter": 0.88,
        },
        "shorts": {
            "spring": 0.34,
            "summer": 0.96,
            "autumn": 0.18,
            "winter": 0.02,
        },
        "mini_skirt": {
            "spring": 0.72,
            "summer": 0.66,
            "autumn": 0.38,
            "winter": 0.08,
        },
        "midi_skirt": {
            "spring": 0.58,
            "summer": 0.42,
            "autumn": 0.68,
            "winter": 0.24,
        },
        "mini_dress": {
            "spring": 0.60,
            "summer": 0.82,
            "autumn": 0.32,
            "winter": 0.06,
        },
        "midi_dress": {
            "spring": 0.56,
            "summer": 0.58,
            "autumn": 0.56,
            "winter": 0.16,
        },
        "sneakers": {
            "spring": 0.66,
            "summer": 0.46,
            "autumn": 0.68,
            "winter": 0.26,
        },
        "boots": {
            "spring": 0.18,
            "summer": 0.02,
            "autumn": 0.68,
            "winter": 0.94,
        },
        "sandals": {
            "spring": 0.28,
            "summer": 0.96,
            "autumn": 0.08,
            "winter": 0.01,
        },
        "heels": {
            "spring": 0.42,
            "summer": 0.52,
            "autumn": 0.42,
            "winter": 0.12,
        },
        "flats": {
            "spring": 0.62,
            "summer": 0.46,
            "autumn": 0.40,
            "winter": 0.12,
        },
        "bucket_hat": {
            "spring": 0.48,
            "summer": 0.64,
            "autumn": 0.20,
            "winter": 0.06,
        },
        "beanie": {
            "spring": 0.12,
            "summer": 0.01,
            "autumn": 0.46,
            "winter": 0.92,
        },
        "hat": {
            "spring": 0.44,
            "summer": 0.62,
            "autumn": 0.24,
            "winter": 0.08,
        },
    }

    score_map = fine_overrides.get(fine_category, base_scores)
    candidates, normalized_map = build_candidates(score_map, label_map)
    selection = _build_single_or_conditional_double_selection(
        candidates,
        second_score_min=0.30,
        max_gap=0.10,
        allowed_pairs=SEASON_ALLOWED_DOUBLE_COMBINATIONS,
        force_single_keys={"summer", "winter"},
    )

    return {
        "selected": selection["selected"],
        "candidates": candidates,
        "candidateScoreMap": normalized_map,
        "selectionDebug": selection,
    }
