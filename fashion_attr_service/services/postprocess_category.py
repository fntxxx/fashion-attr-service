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


def _apply_coarse_type_lock(
    category_result: Dict[str, Any],
    coarse_type: Optional[str],
    category_key: str,
) -> Tuple[Dict[str, Any], str, str]:
    coarse = (coarse_type or "").strip().lower()
    key = category_key

    if coarse == "upper_body":
        key = _normalize_category_key("upper_body", key)
        return (
            _force_main_category(
                category_result,
                "upper_body",
                "上身",
                key,
                UPPER_BODY_CATEGORY_MAP[key],
            ),
            "upper_body",
            key,
        )

    if coarse == "pants":
        key = _normalize_category_key("pants", key)
        return (
            _force_main_category(
                category_result,
                "pants",
                "褲子",
                key,
                PANTS_CATEGORY_MAP[key],
            ),
            "pants",
            key,
        )

    if coarse == "skirt":
        key = _normalize_category_key("skirt", key)
        return (
            _force_main_category(
                category_result,
                "skirt",
                "裙子",
                key,
                SKIRT_CATEGORY_MAP[key],
            ),
            "skirt",
            key,
        )

    if coarse == "dress":
        key = _normalize_category_key("dress", key)
        return (
            _force_main_category(
                category_result,
                "dress",
                "連身裙",
                key,
                DRESS_CATEGORY_MAP[key],
            ),
            "dress",
            key,
        )

    if coarse == "shoes":
        key = _normalize_category_key("shoes", key)
        return (
            _force_main_category(
                category_result,
                "shoes",
                "鞋子",
                key,
                SHOES_CATEGORY_MAP[key],
            ),
            "shoes",
            key,
        )

    if coarse == "headwear":
        key = _normalize_category_key("headwear", key)
        return (
            _force_main_category(
                category_result,
                "headwear",
                "帽子",
                key,
                HEADWEAR_CATEGORY_MAP[key],
            ),
            "headwear",
            key,
        )

    return category_result, category_result["mainCategoryKey"], category_result["categoryKey"]


def _refine_upper_body_key(
    category_key: str,
    validation_best_label: str,
) -> str:
    key = (category_key or "").strip().lower()
    best_label = (validation_best_label or "").strip().lower()

    # cardigan 在目前資料定義中一律視為 outer
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
    main_key = category_result["mainCategoryKey"]
    category_key = category_result["categoryKey"]

    coarse_type = None
    if coarse_info:
        coarse_type = coarse_info.get("coarse_type")

    validation_best_label = ""
    if validation:
        validation_best_label = str(validation.get("best_label", "")).lower()

    # 1. 先把 category key 正規化，避免奇怪 key 滲透到後面
    category_key = _normalize_category_key(main_key, category_key)
    category_result["categoryKey"] = category_key

    if main_key == "upper_body":
        category_result["category"] = UPPER_BODY_CATEGORY_MAP[category_key]
    elif main_key == "pants":
        category_result["category"] = PANTS_CATEGORY_MAP[category_key]
    elif main_key == "skirt":
        category_result["category"] = SKIRT_CATEGORY_MAP[category_key]
    elif main_key == "dress":
        category_result["category"] = DRESS_CATEGORY_MAP[category_key]
    elif main_key == "shoes":
        category_result["category"] = SHOES_CATEGORY_MAP[category_key]
    elif main_key == "headwear":
        category_result["category"] = HEADWEAR_CATEGORY_MAP[category_key]

    # 2. coarse type 優先鎖主類別
    # 這一步直接解決：
    # - skirt -> pants
    # - skirt -> dress
    # - dress -> skirt / pants
    category_result, main_key, category_key = _apply_coarse_type_lock(
        category_result=category_result,
        coarse_type=coarse_type,
        category_key=category_key,
    )

    # 3. upper_body 再做 top / outer 邊界修正
    # 只在 upper_body 類內細分，不讓 rule 跨類互打
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

    # 4. shape heuristic 只留在 skirt 類內做 very-light 補助
    # 不再用它把 skirt 翻成 pants，避免 denim skirt 誤判
    if route == "product" and main_key == "skirt":
        shape_guess = estimate_pants_vs_skirt(image)

        # 如果真的偏裙裝，但 key 很怪，收斂成 midi_skirt
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

    # 最終保護：如果 coarse 已經明確是 skirt，就不要輸出 dress
    if coarse_type == "skirt" and category_result.get("mainCategoryKey") == "dress":
        category_result = _force_main_category(
            category_result,
            "skirt",
            "裙子",
            "midi_skirt",
            "中長裙",
        )

    return category_result