from services.shape_heuristics import estimate_pants_vs_skirt


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


def _force_main_category(category_result, main_key, main_label, category_key, category_label):
    category_result["mainCategoryKey"] = main_key
    category_result["mainCategory"] = main_label
    category_result["categoryKey"] = category_key
    category_result["category"] = category_label
    return category_result


def postprocess_category(category_result, image, color_tone=None, route=None, coarse_info=None, validation=None):
    main_key = category_result["mainCategoryKey"]
    category_key = category_result["categoryKey"]

    coarse_type = None
    if coarse_info:
        coarse_type = coarse_info.get("coarse_type")

    validation_best_label = ""
    if validation:
        validation_best_label = str(validation.get("best_label", "")).lower()

    # 1. Demo 穩定版收斂：裙子與洋裝先保守處理
    if main_key == "skirt" and category_key not in SKIRT_CATEGORY_MAP:
        category_result["categoryKey"] = "midi_skirt"
        category_result["category"] = "中長裙"
        category_key = "midi_skirt"

    if main_key == "dress" and category_key not in DRESS_CATEGORY_MAP:
        category_result["categoryKey"] = "midi_dress"
        category_result["category"] = "中長洋裝"
        category_key = "midi_dress"

    # 2. product 圖優先修正高混淆 case
    if route == "product":
        # validation 明確比較像 pants，但主分類卻是 skirt/mini_skirt
        if (
            main_key == "skirt"
            and category_key == "mini_skirt"
            and "pants" in validation_best_label
        ):
            return _force_main_category(
                category_result,
                "pants",
                "褲子",
                "trousers",
                "長褲",
            )

        # coarse type 作為次要校正：pants / skirt
        if main_key == "skirt" and coarse_type == "pants":
            category_result = _force_main_category(
                category_result,
                "pants",
                "褲子",
                "trousers",
                "長褲",
            )
            main_key = "pants"
            category_key = "trousers"

        elif main_key == "pants" and coarse_type == "skirt":
            category_result = _force_main_category(
                category_result,
                "skirt",
                "裙子",
                "midi_skirt",
                "中長裙",
            )
            main_key = "skirt"
            category_key = "midi_skirt"

    # 3. shape heuristic 仍保留，但只當最後輔助
    shape_guess = estimate_pants_vs_skirt(image)

    if route == "product" and main_key == "skirt" and shape_guess == "pants":
        category_result = _force_main_category(
            category_result,
            "pants",
            "褲子",
            "trousers",
            "長褲",
        )
        main_key = "pants"
        category_key = "trousers"

    # 4. coarse type 與 main category 的一致性約束
    if coarse_type == "upper_body" and main_key in {"pants", "skirt", "dress"}:
        fallback_key = category_key if category_key in UPPER_BODY_CATEGORY_MAP else "shirt"
        category_result = _force_main_category(
            category_result,
            "upper_body",
            "上身",
            fallback_key,
            UPPER_BODY_CATEGORY_MAP[fallback_key],
        )

    elif coarse_type == "pants" and main_key in {"skirt", "dress", "upper_body"}:
        fallback_key = category_key if category_key in PANTS_CATEGORY_MAP else "trousers"
        category_result = _force_main_category(
            category_result,
            "pants",
            "褲子",
            fallback_key,
            PANTS_CATEGORY_MAP[fallback_key],
        )

    elif coarse_type == "skirt" and main_key in {"pants", "dress", "upper_body"}:
        fallback_key = category_key if category_key in SKIRT_CATEGORY_MAP else "midi_skirt"
        category_result = _force_main_category(
            category_result,
            "skirt",
            "裙子",
            fallback_key,
            SKIRT_CATEGORY_MAP[fallback_key],
        )

    elif coarse_type == "dress" and main_key in {"pants", "skirt", "upper_body"}:
        fallback_key = category_key if category_key in DRESS_CATEGORY_MAP else "midi_dress"
        category_result = _force_main_category(
            category_result,
            "dress",
            "連身裙",
            fallback_key,
            DRESS_CATEGORY_MAP[fallback_key],
        )

    elif coarse_type == "shoes" and main_key != "shoes":
        fallback_key = category_key if category_key in SHOES_CATEGORY_MAP else "sneakers"
        category_result = _force_main_category(
            category_result,
            "shoes",
            "鞋子",
            fallback_key,
            SHOES_CATEGORY_MAP[fallback_key],
        )

    elif coarse_type == "headwear" and main_key != "headwear":
        fallback_key = category_key if category_key in HEADWEAR_CATEGORY_MAP else "hat"
        category_result = _force_main_category(
            category_result,
            "headwear",
            "帽子",
            fallback_key,
            HEADWEAR_CATEGORY_MAP[fallback_key],
        )

    return category_result