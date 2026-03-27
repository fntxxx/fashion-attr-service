from typing import Dict

from models.clip_model import (
    score_texts_with_image_feature,
    encode_image_feature,
)
from utils.scoring import build_candidates


STAGE1_LABELS = {
    "headwear": "a clean product photo of a single hat or other headwear",
    "upper_body": "a clean product photo of a single upper-body garment",
    "pants": "a clean product photo of a single pair of pants or trousers",
    "skirt": "a clean product photo of a single skirt",
    "dress": "a clean product photo of a single dress",
    "shoes": "a clean product photo of a single pair of shoes",
}

STAGE2_LABELS = {
    "headwear": {
        "bucket_hat": "a clean product photo of a single bucket hat",
        "beanie": "a clean product photo of a single beanie",
        "hat": "a clean product photo of a single fashion hat or cap",
    },
    "upper_body": {
        "t_shirt": "a clean product photo of a single t-shirt or casual tee",
        "shirt": "a clean product photo of a single button-up shirt or blouse",
        "tank_top": "a clean product photo of a single tank top or sleeveless top",
        "hoodie": "a clean product photo of a single hoodie",
        "sweatshirt": "a clean product photo of a single sweatshirt",
        "knit_sweater": "a clean product photo of a single knit sweater",
        "cardigan": "a clean product photo of a single cardigan",
        "denim_jacket": "a clean product photo of a single denim jacket",
        "blazer": "a clean product photo of a single blazer",
        "coat": "a clean product photo of a single coat or outerwear jacket",
        "puffer_jacket": "a clean product photo of a single puffer jacket",
        "vest": "a clean product photo of a single vest",
        "windbreaker": "a clean product photo of a single windbreaker",
    },
    "pants": {
        "jeans": "a clean product photo of a single pair of jeans",
        "trousers": "a clean product photo of a single pair of trousers",
        "wide_leg_pants": "a clean product photo of a single pair of wide-leg pants",
        "leggings": "a clean product photo of a single pair of leggings",
        "shorts": "a clean product photo of a single pair of shorts",
    },
    "skirt": {
        "mini_skirt": "a clean product photo of a single mini skirt",
        "midi_skirt": "a clean product photo of a single midi skirt or long skirt",
    },
    "dress": {
        "mini_dress": "a clean product photo of a single mini dress",
        "midi_dress": "a clean product photo of a single midi dress or long dress",
    },
    "shoes": {
        "sneakers": "a clean product photo of a single pair of sneakers or casual shoes",
        "boots": "a clean product photo of a single pair of boots",
        "sandals": "a clean product photo of a single pair of sandals",
        "heels": "a clean product photo of a single pair of high heels",
        "flats": "a clean product photo of a single pair of flats",
    },
}

MAIN_CATEGORY_LABEL_MAP = {
    "headwear": "帽子",
    "upper_body": "上身",
    "pants": "褲子",
    "skirt": "裙子",
    "dress": "連身裙",
    "shoes": "鞋子",
}

FINE_CATEGORY_LABEL_MAP = {
    "bucket_hat": "漁夫帽",
    "beanie": "毛帽",
    "hat": "帽子",
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
    "jeans": "牛仔褲",
    "trousers": "長褲",
    "wide_leg_pants": "寬褲",
    "leggings": "內搭褲",
    "shorts": "短褲",
    "mini_skirt": "短裙",
    "midi_skirt": "中長裙",
    "mini_dress": "短洋裝",
    "midi_dress": "中長洋裝",
    "sneakers": "休閒鞋",
    "boots": "靴子",
    "sandals": "涼鞋",
    "heels": "高跟鞋",
    "flats": "平底鞋",
}


def _score_label_map_with_confidence(
    image,
    prompt_map: Dict[str, str],
    display_label_map: Dict[str, str],
    image_features=None,
    model_backend: str | None = None,
):
    keys = list(prompt_map.keys())
    prompts = list(prompt_map.values())

    if image_features is None:
        image_features = encode_image_feature(image, model_backend=model_backend)

    raw_results = score_texts_with_image_feature(image_features, prompts, model_backend=model_backend)
    raw_score_map = {
        key: float(raw_results[idx]["score"])
        for idx, key in enumerate(keys)
    }

    candidates, normalized_map = build_candidates(
        raw_score_map,
        {key: display_label_map[key] for key in keys},
    )

    best = candidates[0]
    return {
        "best_key": str(best["value"]),
        "best_score": float(best["score"]),
        "candidates": candidates,
        "score_map": normalized_map,
        "raw_score_map": raw_score_map,
    }


def classify_category(image, image_features=None, model_backend: str | None = None):
    if image_features is None:
        image_features = encode_image_feature(image, model_backend=model_backend)

    main_result = _score_label_map_with_confidence(
        image,
        STAGE1_LABELS,
        MAIN_CATEGORY_LABEL_MAP,
        image_features=image_features,
        model_backend=model_backend,
    )
    main_key = main_result["best_key"]

    fine_result = _score_label_map_with_confidence(
        image,
        STAGE2_LABELS[main_key],
        FINE_CATEGORY_LABEL_MAP,
        image_features=image_features,
        model_backend=model_backend,
    )
    fine_key = fine_result["best_key"]

    return {
        "mainCategoryKey": main_key,
        "mainCategory": MAIN_CATEGORY_LABEL_MAP[main_key],
        "categoryKey": fine_key,
        "category": FINE_CATEGORY_LABEL_MAP[fine_key],
        "scores": {
            "mainCategory": float(main_result["best_score"]),
            "category": float(fine_result["best_score"]),
        },
        "score": float(fine_result["best_score"]),
        "candidates": {
            "mainCategory": main_result["candidates"],
            "category": fine_result["candidates"],
        },
        "candidateScoreMaps": {
            "mainCategory": main_result["score_map"],
            "category": fine_result["score_map"],
        },
    }