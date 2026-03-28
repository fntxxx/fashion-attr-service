from typing import Any, Optional, Dict, Tuple

from fashion_attr_service.services.shape_heuristics import estimate_pants_vs_skirt


UPPER_BODY_CATEGORY_MAP = {
    "t_shirt": "T 恤",
    "shirt": "襯衫",
    "tank_top": "背心",
    "hoodie": "帽T",
    "sweatshirt": "大學T",
    "knit_sweater": "針織衫",
    "cardigan": "開襟衫",
    "denim_jacket": "牛仔外套",
    "blazer": "西裝外套",
    "coat": "外套",
    "puffer_jacket": "鋪棉外套",
    "vest": "背心外套",
    "windbreaker": "防風外套",
}

PANTS_CATEGORY_MAP = {
    "jeans": "牛仔褲",
    "trousers": "長褲",
    "wide_leg_pants": "寬褲",
    "leggings": "內搭褲",
    "shorts": "短褲",
}

SKIRT_CATEGORY_MAP = {
    "mini_skirt": "短裙",
    "midi_skirt": "中長裙",
}

DRESS_CATEGORY_MAP = {
    "mini_dress": "短洋裝",
    "midi_dress": "中長洋裝",
}

SHOES_CATEGORY_MAP = {
    "sneakers": "休閒鞋",
    "boots": "靴子",
    "sandals": "涼鞋",
    "heels": "高跟鞋",
    "flats": "平底鞋",
}

HEADWEAR_CATEGORY_MAP = {
    "bucket_hat": "漁夫帽",
    "beanie": "毛帽",
    "hat": "帽子",
}

OUTER_FINE_KEYS = {
    "blazer",
    "coat",
    "puffer_jacket",
    "vest",
    "windbreaker",
    "denim_jacket",
    "cardigan",
}

TOP_FINE_KEYS = {
    "t_shirt",
    "shirt",
    "tank_top",
    "hoodie",
    "sweatshirt",
    "knit_sweater",
}

HIGH_RISK_COARSE_TYPES = {"upper_body", "dress", "skirt"}


def _force_main_category(
    category_result: Dict[str, Any],
    main_key: str,
    main_label: str,
    category_key: str,
    category_label: str,
) -> Dict[str, Any]:
    category_result["mainCategoryKey"] = main_key
    category_result["mainCategory"] = main_label
    category_result["categoryKey"] = category_key
    category_result["category"] = category_label
    return category_result


def _normalize_category_key(
    main_key: str,
    category_key: str,
) -> str:
    if main_key == "upper_body" and category_key not in UPPER_BODY_CATEGORY_MAP:
        return "shirt"
    if main_key == "pants" and category_key not in PANTS_CATEGORY_MAP:
        return "trousers"
    if main_key == "skirt" and category_key not in SKIRT_CATEGORY_MAP:
        return "midi_skirt"
    if main_key == "dress" and category_key not in DRESS_CATEGORY_MAP:
        return "midi_dress"
    if main_key == "shoes" and category_key not in SHOES_CATEGORY_MAP:
        return "sneakers"
    if main_key == "headwear" and category_key not in HEADWEAR_CATEGORY_MAP:
        return "hat"
    return category_key


def _get_main_category_label(main_key: str) -> str:
    return {
        "upper_body": "上身",
        "pants": "褲子",
        "skirt": "裙子",
        "dress": "連身裙",
        "shoes": "鞋子",
        "headwear": "帽子",
    }[main_key]


def _get_fine_category_label(main_key: str, category_key: str) -> str:
    if main_key == "upper_body":
        return UPPER_BODY_CATEGORY_MAP[category_key]
    if main_key == "pants":
        return PANTS_CATEGORY_MAP[category_key]
    if main_key == "skirt":
        return SKIRT_CATEGORY_MAP[category_key]
    if main_key == "dress":
        return DRESS_CATEGORY_MAP[category_key]
    if main_key == "shoes":
        return SHOES_CATEGORY_MAP[category_key]
    if main_key == "headwear":
        return HEADWEAR_CATEGORY_MAP[category_key]
    return category_key


def _read_main_score(category_result: Dict[str, Any], main_key: str) -> float:
    candidate_score_maps = category_result.get("candidateScoreMaps", {})
    main_score_map = candidate_score_maps.get("mainCategory", {})
    if main_key in main_score_map:
        return float(main_score_map[main_key])
    return float(category_result.get("scores", {}).get("mainCategory", 0.0))


def _should_apply_coarse_override(
    *,
    coarse_type: str,
    current_main_key: str,
    coarse_score: float,
    current_main_score: float,
) -> tuple[bool, dict[str, Any]]:
    if not coarse_type or coarse_type == current_main_key:
        return False, {
            "reason": "same_or_missing_coarse",
            "required_margin": 0.0,
            "score_gap": coarse_score - current_main_score,
        }

    required_margin = 0.08
    if coarse_type in HIGH_RISK_COARSE_TYPES or current_main_key in HIGH_RISK_COARSE_TYPES:
        required_margin = 0.15

    if coarse_type == "upper_body" and current_main_key == "upper_body":
        required_margin = 0.0

    score_gap = coarse_score - current_main_score
    should_apply = score_gap >= required_margin
    return should_apply, {
        "reason": "override" if should_apply else "insufficient_margin",
        "required_margin": required_margin,
        "score_gap": score_gap,
    }


def _apply_coarse_type_lock(
    category_result: Dict[str, Any],
    coarse_info: Optional[Dict[str, Any]],
    category_key: str,
) -> Tuple[Dict[str, Any], str, str]:
    coarse_info = coarse_info or {}
    coarse = str(coarse_info.get("coarse_type") or "").strip().lower()
    key = category_key
    current_main_key = str(category_result.get("mainCategoryKey") or "")
    coarse_score = float(coarse_info.get("score") or 0.0)
    current_main_score = _read_main_score(category_result, current_main_key)

    should_apply, decision_meta = _should_apply_coarse_override(
        coarse_type=coarse,
        current_main_key=current_main_key,
        coarse_score=coarse_score,
        current_main_score=current_main_score,
    )

    debug_info = category_result.setdefault("postprocessDebug", {})
    debug_info["coarse_decision"] = {
        "coarse_type": coarse,
        "coarse_score": coarse_score,
        "current_main_key": current_main_key,
        "current_main_score": current_main_score,
        **decision_meta,
    }

    if not should_apply:
        return category_result, category_result["mainCategoryKey"], category_result["categoryKey"]

    if coarse not in {"upper_body", "pants", "skirt", "dress", "shoes", "headwear"}:
        return category_result, category_result["mainCategoryKey"], category_result["categoryKey"]

    key = _normalize_category_key(coarse, key)
    return (
        _force_main_category(
            category_result,
            coarse,
            _get_main_category_label(coarse),
            key,
            _get_fine_category_label(coarse, key),
        ),
        coarse,
        key,
    )


def _refine_upper_body_key(
    category_key: str,
    validation_best_label: str,
) -> str:
    key = (category_key or "").strip().lower()
    best_label = (validation_best_label or "").strip().lower()

    if key == "cardigan":
        return "cardigan"

    if "jacket" in best_label:
        return "denim_jacket"
    if "coat" in best_label:
        return "coat"

    if key in OUTER_FINE_KEYS:
        return key

    if key in TOP_FINE_KEYS:
        return key

    if "shirt" in best_label:
        return "shirt"

    return "shirt"


def postprocess_category(
    category_result,
    image,
    color_tone=None,
    route=None,
    coarse_info=None,
    validation=None,
):
    original_main_key = category_result["mainCategoryKey"]
    original_category_key = category_result["categoryKey"]

    validation_best_label = ""
    if validation:
        validation_best_label = str(validation.get("best_label", "")).lower()

    category_result["postprocessDebug"] = {
        "pre_postprocess": {
            "mainCategoryKey": original_main_key,
            "categoryKey": original_category_key,
            "mainCategory": category_result.get("mainCategory"),
            "category": category_result.get("category"),
        }
    }

    main_key = original_main_key
    category_key = original_category_key

    category_key = _normalize_category_key(main_key, category_key)
    category_result["categoryKey"] = category_key
    category_result["category"] = _get_fine_category_label(main_key, category_key)

    category_result, main_key, category_key = _apply_coarse_type_lock(
        category_result=category_result,
        coarse_info=coarse_info,
        category_key=category_key,
    )

    if main_key == "upper_body":
        refined_key = _refine_upper_body_key(
            category_key=category_key,
            validation_best_label=validation_best_label,
        )

        category_result = _force_main_category(
            category_result,
            "upper_body",
            "上身",
            refined_key,
            UPPER_BODY_CATEGORY_MAP[refined_key],
        )
        main_key = "upper_body"
        category_key = refined_key

    if route == "product" and main_key == "skirt":
        shape_guess = estimate_pants_vs_skirt(image)
        if shape_guess == "skirt" and category_key not in SKIRT_CATEGORY_MAP:
            category_result = _force_main_category(
                category_result,
                "skirt",
                "裙子",
                "midi_skirt",
                "中長裙",
            )
            main_key = "skirt"
            category_key = "midi_skirt"

    coarse_type = str((coarse_info or {}).get("coarse_type") or "").strip().lower()
    if coarse_type == "skirt" and category_result.get("mainCategoryKey") == "dress":
        coarse_score = float((coarse_info or {}).get("score") or 0.0)
        current_main_score = _read_main_score(category_result, "dress")
        if coarse_score - current_main_score >= 0.15:
            category_result = _force_main_category(
                category_result,
                "skirt",
                "裙子",
                "midi_skirt",
                "中長裙",
            )
            main_key = "skirt"
            category_key = "midi_skirt"

    category_result["postprocessDebug"]["postprocess"] = {
        "mainCategoryKey": main_key,
        "categoryKey": category_key,
        "mainCategory": category_result.get("mainCategory"),
        "category": category_result.get("category"),
        "changed": (
            original_main_key != category_result.get("mainCategoryKey")
            or original_category_key != category_result.get("categoryKey")
        ),
    }
    return category_result
