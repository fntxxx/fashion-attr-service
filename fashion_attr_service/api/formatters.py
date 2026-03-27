from __future__ import annotations

from fashion_attr_service.api.constants import (
    CATEGORY_UI_OPTIONS,
    CATEGORY_VALUE_TO_LABEL,
    MAIN_CATEGORY_TO_UI,
    OUTER_FINE_KEYS,
)


TOP_FINE_KEYS = {
    "t_shirt",
    "shirt",
    "tank_top",
    "hoodie",
    "sweatshirt",
    "knit_sweater",
}


def build_category_value(category_result: dict) -> str:
    fine_key = category_result["categoryKey"]
    main_key = category_result["mainCategoryKey"]

    if fine_key in OUTER_FINE_KEYS:
        return "outer"

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

    return {
        key: float(value / total)
        for key, value in score_map.items()
    }


def build_category_candidates(category_result: dict) -> list[dict]:
    candidate_score_maps = category_result.get("candidateScoreMaps", {})
    main_score_map = candidate_score_maps.get("mainCategory", {})
    fine_score_map = candidate_score_maps.get("category", {})

    if not main_score_map or not fine_score_map:
        selected = build_category_value(category_result)
        candidates = []
        for value, label in CATEGORY_UI_OPTIONS:
            candidates.append(
                {
                    "value": value,
                    "label": label,
                    "score": 1.0 if value == selected else 0.0,
                }
            )
        return candidates

    upper_body_prob = float(main_score_map.get("upper_body", 0.0))
    headwear_prob = float(main_score_map.get("headwear", 0.0))

    ui_score_map = {
        "top": upper_body_prob * sum(float(fine_score_map.get(key, 0.0)) for key in TOP_FINE_KEYS) + headwear_prob,
        "pants": float(main_score_map.get("pants", 0.0)),
        "skirt": float(main_score_map.get("skirt", 0.0)),
        "dress": float(main_score_map.get("dress", 0.0)),
        "outer": upper_body_prob * sum(float(fine_score_map.get(key, 0.0)) for key in OUTER_FINE_KEYS),
        "shoes": float(main_score_map.get("shoes", 0.0)),
    }

    ui_score_map = _normalize_probability_map(ui_score_map)

    candidates = [
        {
            "value": value,
            "label": label,
            "score": float(ui_score_map.get(value, 0.0)),
        }
        for value, label in CATEGORY_UI_OPTIONS
    ]

    return sorted(candidates, key=lambda item: item["score"], reverse=True)


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
    category_value = build_category_value(category_result)
    color_value = color_payload["color"]
    occasion_values = occasions["selected"]
    season_values = seasons["selected"]

    return {
        "ok": True,
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
            "occasion": float(max([x["score"] for x in occasions["candidates"]] or [0.0])),
            "color": float(max([x["score"] for x in color_payload["candidates"]] or [0.0])),
            "season": float(max([x["score"] for x in seasons["candidates"]] or [0.0])),
        },
        "candidates": {
            "category": build_category_candidates(category_result),
            "color": color_payload["candidates"],
            "occasion": occasions["candidates"],
            "season": seasons["candidates"],
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