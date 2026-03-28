from typing import Dict

from fashion_attr_service.models.fashion_siglip_model import (
    score_texts_with_image_feature,
    encode_image_feature,
)
from fashion_attr_service.utils.scoring import build_candidates


# Prompt 設計原則：
# - 維持所有 prompt 風格一致，避免同一批候選描述長短差異過大。
# - 保留「clean product photo」骨架，補足最必要的服裝輪廓、穿著位置、常見英文別名。
# - stage1 專注在大類分流；stage2 專注在同大類內的細粒度區辨。
# - 本次只做最小範圍微調：
#   1) 強化 dress 與 top 的邊界
#   2) 保留 cardigan 在 outer 側，避免一般 top 被吸到 outer
STAGE1_LABELS = {
    "headwear": "a clean product photo of a single headwear item worn on the head, such as a hat, cap, beanie, or bucket hat",
    "upper_body": "a clean product photo of a single separate upper-body garment worn on the torso, including tops and outerwear such as t-shirts, shirts, sweaters, hoodies, jackets, coats, and vests",
    "pants": "a clean product photo of a single bottom garment with two leg openings, such as pants, trousers, jeans, leggings, or shorts",
    "skirt": "a clean product photo of a single skirt, a bottom garment that hangs from the waist without separate leg openings",
    "dress": "a clean product photo of a single dress, a one-piece garment that covers the upper body and continues below the waist into a skirt",
    "shoes": "a clean product photo of a single pair of footwear, such as sneakers, boots, sandals, heels, or flats",
}

STAGE2_LABELS = {
    "headwear": {
        "bucket_hat": "a clean product photo of a single bucket hat, a soft hat with a wide downward-sloping brim",
        "beanie": "a clean product photo of a single beanie, a close-fitting knit cap without a brim",
        "hat": "a clean product photo of a single fashion hat or cap, a general headwear item other than a bucket hat or beanie",
    },
    "upper_body": {
        "t_shirt": "a clean product photo of a single t-shirt or tee, a separate casual short-sleeve knit top worn as the main top, without buttons or a front opening",
        "shirt": "a clean product photo of a single shirt or blouse, a separate woven upper-body garment worn as the main top, with a collar or button front, not an outerwear layer",
        "tank_top": "a clean product photo of a single tank top or sleeveless top with no sleeves or shoulder coverage",
        "hoodie": "a clean product photo of a single hoodie, a casual hooded sweatshirt made from soft knit fabric",
        "sweatshirt": "a clean product photo of a single sweatshirt, a pullover long-sleeve casual top without a hood or front opening",
        "knit_sweater": "a clean product photo of a single knit sweater or pullover, a knitted long-sleeve top with a soft textured knit look",
        "cardigan": "a clean product photo of a single cardigan, a knit outerwear layering piece with a front opening or buttons, worn over inner clothing",
        "denim_jacket": "a clean product photo of a single denim jacket, a structured outerwear jacket made of blue or black denim",
        "blazer": "a clean product photo of a single blazer, a tailored structured outerwear jacket with lapels for smart or formal wear",
        "coat": "a clean product photo of a single coat, a longer structured or warm outerwear layer worn over other clothing, typically longer than a regular jacket",
        "puffer_jacket": "a clean product photo of a single puffer jacket, a quilted padded jacket with a bulky insulated shape",
        "vest": "a clean product photo of a single vest or gilet, a sleeveless outerwear layering piece worn over other clothing",
        "windbreaker": "a clean product photo of a single windbreaker, a lightweight sporty zip-front jacket for wind or light rain",
    },
    "pants": {
        "jeans": "a clean product photo of a single pair of jeans, denim pants with a classic five-pocket casual style",
        "trousers": "a clean product photo of a single pair of trousers, long tailored pants in woven fabric, not denim",
        "wide_leg_pants": "a clean product photo of a single pair of wide-leg pants, trousers with a loose wide silhouette from hip to hem",
        "leggings": "a clean product photo of a single pair of leggings, tight close-fitting stretch pants that hug the legs",
        "shorts": "a clean product photo of a single pair of shorts, short pants ending above the knee",
    },
    "skirt": {
        "mini_skirt": "a clean product photo of a single mini skirt, a short skirt ending above the knee",
        "midi_skirt": "a clean product photo of a single midi skirt or long skirt, a skirt extending below the knee toward the calf or ankle",
    },
    "dress": {
        "mini_dress": "a clean product photo of a single mini dress, a short one-piece garment that covers the upper body and continues into a skirt above the knee",
        "midi_dress": "a clean product photo of a single midi dress or long dress, a one-piece garment that covers the upper body and continues into a skirt below the knee toward the calf or ankle",
    },
    "shoes": {
        "sneakers": "a clean product photo of a single pair of sneakers, casual athletic lace-up shoes with rubber soles",
        "boots": "a clean product photo of a single pair of boots, shoes that cover the ankle or rise higher on the leg",
        "sandals": "a clean product photo of a single pair of sandals, open-toe shoes with straps and exposed foot areas",
        "heels": "a clean product photo of a single pair of high heels or pumps, dress shoes with a clearly elevated heel",
        "flats": "a clean product photo of a single pair of flats, low-profile closed-toe shoes with little or no heel",
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