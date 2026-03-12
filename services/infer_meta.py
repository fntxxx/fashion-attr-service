from __future__ import annotations

from typing import Any


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


def _build_candidate_list(
    score_map: dict[str, float],
    label_map: dict[str, str],
) -> list[dict[str, Any]]:
    items = []
    for key, label in label_map.items():
        items.append({
            "value": key,
            "label": label,
            "score": float(score_map.get(key, 0.0)),
        })
    return sorted(items, key=lambda x: x["score"], reverse=True)


def _pick_multi_selected(
    candidates: list[dict[str, Any]],
    threshold: float,
    max_selected: int,
) -> list[str]:
    selected = [
        item["value"]
        for item in candidates
        if float(item["score"]) >= threshold
    ][:max_selected]

    if selected:
        return selected

    return [candidates[0]["value"]] if candidates else []


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
            "campus_casual": 0.55,
            "social": 0.50,
            "business_casual": 0.68,
            "professional": 0.44,
        },
        "denim_jacket": {
            "campus_casual": 0.82,
            "social": 0.42,
            "business_casual": 0.18,
            "professional": 0.08,
        },
        "blazer": {
            "campus_casual": 0.20,
            "social": 0.50,
            "business_casual": 0.84,
            "professional": 0.90,
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
            "campus_casual": 0.36,
            "social": 0.44,
            "business_casual": 0.84,
            "professional": 0.74,
        },
        "wide_leg_pants": {
            "campus_casual": 0.58,
            "social": 0.56,
            "business_casual": 0.54,
            "professional": 0.28,
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
            "campus_casual": 0.46,
            "social": 0.54,
            "business_casual": 0.46,
            "professional": 0.20,
        },
        "sandals": {
            "campus_casual": 0.72,
            "social": 0.34,
            "business_casual": 0.08,
            "professional": 0.04,
        },
        "heels": {
            "campus_casual": 0.08,
            "social": 0.72,
            "business_casual": 0.52,
            "professional": 0.44,
        },
        "flats": {
            "campus_casual": 0.34,
            "social": 0.42,
            "business_casual": 0.62,
            "professional": 0.48,
        },
        "bucket_hat": {
            "campus_casual": 0.72,
            "social": 0.22,
            "business_casual": 0.04,
            "professional": 0.02,
        },
        "beanie": {
            "campus_casual": 0.68,
            "social": 0.20,
            "business_casual": 0.04,
            "professional": 0.02,
        },
        "hat": {
            "campus_casual": 0.50,
            "social": 0.40,
            "business_casual": 0.12,
            "professional": 0.06,
        },
    }

    score_map = fine_overrides.get(fine_category, base_scores)
    candidates = _build_candidate_list(score_map, label_map)
    selected = _pick_multi_selected(candidates, threshold=0.62, max_selected=2)

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
            "summer": 0.38,
            "autumn": 0.74,
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
            "spring": 0.58,
            "summer": 0.10,
            "autumn": 0.80,
            "winter": 0.62,
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
            "spring": 0.18,
            "summer": 0.02,
            "autumn": 0.62,
            "winter": 0.94,
        },
        "puffer_jacket": {
            "spring": 0.08,
            "summer": 0.01,
            "autumn": 0.42,
            "winter": 0.98,
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
            "spring": 0.62,
            "summer": 0.74,
            "autumn": 0.42,
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
            "summer": 0.52,
            "autumn": 0.62,
            "winter": 0.18,
        },
        "bucket_hat": {
            "spring": 0.46,
            "summer": 0.72,
            "autumn": 0.26,
            "winter": 0.06,
        },
        "beanie": {
            "spring": 0.08,
            "summer": 0.01,
            "autumn": 0.44,
            "winter": 0.96,
        },
        "hat": {
            "spring": 0.44,
            "summer": 0.58,
            "autumn": 0.32,
            "winter": 0.12,
        },
    }

    score_map = fine_overrides.get(fine_category, base_scores)
    candidates = _build_candidate_list(score_map, label_map)
    selected = _pick_multi_selected(candidates, threshold=0.58, max_selected=2)

    if selected == ["summer"]:
        legacy_season = "summer"
    elif selected == ["winter"]:
        legacy_season = "winter"
    elif set(selected) == {"spring", "autumn"}:
        legacy_season = "spring_autumn"
    elif len(selected) >= 3:
        legacy_season = "all_season"
    else:
        legacy_season = "spring_autumn"

    return {
        "selected": selected,
        "candidates": candidates,
        "legacy_season": legacy_season,
    }


def infer_style(main_category: str, fine_category: str) -> str:
    return infer_occasions(main_category, fine_category)["legacy_style"]


def infer_season(main_category: str, fine_category: str) -> str:
    return infer_seasons(main_category, fine_category)["legacy_season"]