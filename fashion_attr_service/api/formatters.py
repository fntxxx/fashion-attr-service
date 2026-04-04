from __future__ import annotations

from fashion_attr_service.api.constants import (
    CATEGORY_UI_OPTIONS,
    CATEGORY_VALUE_TO_LABEL,
    MAIN_CATEGORY_TO_UI,
    OUTER_FINE_KEYS,
    PUBLIC_COLOR_VALUE_MAP,
    PUBLIC_OCCASION_VALUE_MAP,
    PUBLIC_SEASON_VALUE_MAP,
)
from fashion_attr_service.utils.category_catalog import TOP_FINE_KEYS


CARDIGAN_TOP_SCORE_MIN = 0.91
CARDIGAN_OUTER_SUPPORT_MAX = 0.05
CARDIGAN_TOP_SUPPORT_MIN = 0.025
DRESS_TO_OUTER_UPPER_BODY_MIN = 0.30
DRESS_TO_OUTER_MAX_GAP = 0.03
DRESS_TO_OUTER_CATEGORY_SCORE_MAX = 0.75


def _sum_scores(score_map: dict[str, float], keys: set[str]) -> float:
    return sum(float(score_map.get(key, 0.0)) for key in keys)



def _should_map_cardigan_to_top(category_result: dict) -> bool:
    fine_score_map = category_result.get("candidateScoreMaps", {}).get("category", {})
    if not fine_score_map:
        return False

    cardigan_score = float(fine_score_map.get("cardigan", 0.0))
    outer_support = _sum_scores(fine_score_map, OUTER_FINE_KEYS - {"cardigan"})
    top_support = _sum_scores(fine_score_map, TOP_FINE_KEYS)

    return (
        cardigan_score >= CARDIGAN_TOP_SCORE_MIN
        and outer_support <= CARDIGAN_OUTER_SUPPORT_MAX
        and top_support >= CARDIGAN_TOP_SUPPORT_MIN
    )



def _should_map_dress_result_to_outer(category_result: dict, coarse_type: str) -> bool:
    if coarse_type != "upper_body":
        return False
    if category_result.get("mainCategoryKey") != "dress":
        return False

    main_score_map = category_result.get("candidateScoreMaps", {}).get("mainCategory", {})
    if not main_score_map:
        return False

    upper_body_score = float(main_score_map.get("upper_body", 0.0))
    dress_score = float(main_score_map.get("dress", 0.0))
    category_score = float(category_result.get("scores", {}).get("category", 0.0))

    return (
        upper_body_score >= DRESS_TO_OUTER_UPPER_BODY_MIN
        and dress_score - upper_body_score <= DRESS_TO_OUTER_MAX_GAP
        and category_score <= DRESS_TO_OUTER_CATEGORY_SCORE_MAX
    )



def build_category_value(category_result: dict, coarse_type: str = "") -> str:
    fine_key = category_result["categoryKey"]
    main_key = category_result["mainCategoryKey"]

    if fine_key == "cardigan":
        return "top" if _should_map_cardigan_to_top(category_result) else "outerwear"

    if fine_key in OUTER_FINE_KEYS:
        return "outerwear"

    if _should_map_dress_result_to_outer(category_result, coarse_type):
        return "outerwear"

    return MAIN_CATEGORY_TO_UI.get(main_key, "top")



def build_name_value(category_result: dict) -> str:
    fine_label = str(category_result.get("category", "")).strip()
    if fine_label:
        return fine_label

    main_label = str(category_result.get("mainCategory", "")).strip()
    if main_label:
        return main_label

    return "未命名衣物"



def _normalize_probability_map(score_map: dict[str, float]) -> dict[str, float]:
    total = float(sum(score_map.values()))
    if total <= 0:
        return {key: 0.0 for key in score_map.keys()}

    return {key: float(value / total) for key, value in score_map.items()}



def _swap_selected_category_to_top(ui_score_map: dict[str, float], selected_value: str) -> dict[str, float]:
    if selected_value not in ui_score_map:
        return _normalize_probability_map(ui_score_map)

    current_top_value = max(ui_score_map, key=ui_score_map.get)
    if current_top_value == selected_value:
        return _normalize_probability_map(ui_score_map)

    swapped = dict(ui_score_map)
    swapped[selected_value], swapped[current_top_value] = swapped[current_top_value], swapped[selected_value]
    return _normalize_probability_map(swapped)



def build_category_candidates(category_result: dict, coarse_type: str = "") -> list[dict]:
    candidate_score_maps = category_result.get("candidateScoreMaps", {})
    main_score_map = candidate_score_maps.get("mainCategory", {})
    fine_score_map = candidate_score_maps.get("category", {})

    if not main_score_map or not fine_score_map:
        selected = build_category_value(category_result, coarse_type=coarse_type)
        return _build_single_selected_candidate(selected)

    upper_body_prob = float(main_score_map.get("upper_body", 0.0))
    headwear_prob = float(main_score_map.get("headwear", 0.0))

    ui_score_map = {
        "top": upper_body_prob * sum(float(fine_score_map.get(key, 0.0)) for key in TOP_FINE_KEYS) + headwear_prob,
        "bottom": float(main_score_map.get("pants", 0.0)),
        "outerwear": upper_body_prob * sum(float(fine_score_map.get(key, 0.0)) for key in OUTER_FINE_KEYS),
        "shoes": float(main_score_map.get("shoes", 0.0)),
        "skirt": float(main_score_map.get("skirt", 0.0)),
        "dress": float(main_score_map.get("dress", 0.0)),
    }

    if _should_map_dress_result_to_outer(category_result, coarse_type):
        ui_score_map["outerwear"] = max(ui_score_map["outerwear"], upper_body_prob)

    selected = build_category_value(category_result, coarse_type=coarse_type)
    ui_score_map = _swap_selected_category_to_top(ui_score_map, selected)

    candidates = [
        {
            "value": value,
            "label": label,
            "score": float(ui_score_map.get(value, 0.0)),
        }
        for value, label in CATEGORY_UI_OPTIONS
    ]

    return sorted(candidates, key=lambda item: item["score"], reverse=True)



def _build_single_selected_candidate(selected: str) -> list[dict]:
    return [
        {
            "value": value,
            "label": label,
            "score": 1.0 if value == selected else 0.0,
        }
        for value, label in CATEGORY_UI_OPTIONS
    ]



def map_public_value(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def map_public_selected(values: list[str], mapping: dict[str, str]) -> list[str]:
    return [map_public_value(str(value), mapping) for value in values]


def map_public_candidates(candidates: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [
        {
            **candidate,
            "value": map_public_value(str(candidate.get("value", "")), mapping),
        }
        for candidate in candidates
    ]


def _top_candidate_score(candidates: list[dict]) -> float:
    return float(max((float(item["score"]) for item in candidates), default=0.0))


def build_predict_payload(
    *,
    route: str,
    coarse_type: str,
    category_result: dict,
    color_payload: dict,
    occasions: dict,
    seasons: dict,
    validation: dict,
    detection: dict,
    final_score: float,
) -> dict:
    category_value = build_category_value(category_result, coarse_type=coarse_type)
    color_value = map_public_value(str(color_payload["color"]), PUBLIC_COLOR_VALUE_MAP)
    occasion_values = map_public_selected(occasions["selected"], PUBLIC_OCCASION_VALUE_MAP)
    season_values = map_public_selected(seasons["selected"], PUBLIC_SEASON_VALUE_MAP)

    return {
        "route": route,
        "coarseType": coarse_type,
        "name": build_name_value(category_result),
        "category": category_value,
        "categoryLabel": CATEGORY_VALUE_TO_LABEL[category_value],
        "color": color_value,
        "colorLabel": color_payload["colorLabel"],
        "occasion": occasion_values,
        "season": season_values,
        "score": float(final_score),
        "scores": {
            "mainCategory": float(category_result["scores"]["mainCategory"]),
            "category": float(category_result["scores"]["category"]),
            "occasion": _top_candidate_score(occasions["candidates"]),
            "color": _top_candidate_score(color_payload["candidates"]),
            "season": _top_candidate_score(seasons["candidates"]),
        },
        "candidates": {
            "category": build_category_candidates(category_result, coarse_type=coarse_type),
            "color": map_public_candidates(color_payload["candidates"], PUBLIC_COLOR_VALUE_MAP),
            "occasion": map_public_candidates(occasions["candidates"], PUBLIC_OCCASION_VALUE_MAP),
            "season": map_public_candidates(seasons["candidates"], PUBLIC_SEASON_VALUE_MAP),
        },
        "detected": detection["detected"],
        "detectedLabel": detection["label"],
        "bbox": detection["bbox"],
        "validation": {
            "best_label": validation["best_label"],
            "valid_score": validation["valid_score"],
            "invalid_score": validation["invalid_score"],
        },
    }
