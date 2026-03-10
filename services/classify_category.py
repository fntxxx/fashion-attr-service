from models.clip_model import predict_best

STAGE1_LABELS = {
    "headwear": "a clean product photo of a single hat or other headwear",
    "upper_body": "a clean product photo of a single upper-body garment",
    "lower_body": "a clean product photo of a single lower-body garment",
    "dress": "a clean product photo of a single dress",
    "shoes": "a clean product photo of a single pair of shoes"
}

STAGE2_LABELS = {
    "headwear": {
        "baseball_cap": "a clean product photo of a single baseball cap",
        "bucket_hat": "a clean product photo of a single bucket hat",
        "beanie": "a clean product photo of a single beanie",
        "beret": "a clean product photo of a single beret",
        "sun_hat": "a clean product photo of a single sun hat",
        "knit_hat": "a clean product photo of a single knit hat"
    },
    "upper_body": {
        "t_shirt": "a clean product photo of a single t-shirt",
        "graphic_t_shirt": "a clean product photo of a single graphic t-shirt",
        "long_sleeve_t_shirt": "a clean product photo of a single long sleeve t-shirt",
        "polo_shirt": "a clean product photo of a single polo shirt",
        "shirt": "a clean product photo of a single shirt",
        "oxford_shirt": "a clean product photo of a single oxford shirt",
        "blouse": "a clean product photo of a single blouse",
        "tank_top": "a clean product photo of a single tank top",
        "camisole": "a clean product photo of a single camisole",
        "hoodie": "a clean product photo of a single hoodie with a hood",
        "sweatshirt": "a clean product photo of a single crewneck sweatshirt without a hood",
        "knit_sweater": "a clean product photo of a single knit sweater",
        "cardigan": "a clean product photo of a single cardigan",
        "denim_jacket": "a clean product photo of a single denim jacket",
        "bomber_jacket": "a clean product photo of a single bomber jacket",
        "blazer": "a clean product photo of a single blazer",
        "coat": "a clean product photo of a single coat",
        "trench_coat": "a clean product photo of a single trench coat",
        "puffer_jacket": "a clean product photo of a single puffer jacket",
        "vest": "a clean product photo of a single vest",
        "windbreaker": "a clean product photo of a single windbreaker"
    },
    "lower_body": {
        "jeans": "a clean product photo of a single pair of jeans",
        "trousers": "a clean product photo of a single pair of trousers",
        "wide_leg_pants": "a clean product photo of a single pair of wide-leg pants",
        "cargo_pants": "a clean product photo of a single pair of cargo pants",
        "jogger_pants": "a clean product photo of a single pair of jogger pants",
        "leggings": "a clean product photo of a single pair of leggings",
        "shorts": "a clean product photo of a single pair of shorts",
        "denim_shorts": "a clean product photo of a single pair of denim shorts",
        "mini_skirt": "a clean product photo of a single mini skirt",
        "midi_skirt": "a clean product photo of a single midi skirt",
        "pleated_skirt": "a clean product photo of a single pleated skirt",
        "denim_skirt": "a clean product photo of a single denim skirt"
    },
    "dress": {
        "mini_dress": "a clean product photo of a single mini dress",
        "midi_dress": "a clean product photo of a single midi dress",
        "maxi_dress": "a clean product photo of a single maxi dress",
        "shirt_dress": "a clean product photo of a single shirt dress",
        "slip_dress": "a clean product photo of a single slip dress",
        "knit_dress": "a clean product photo of a single knit dress",
        "floral_dress": "a clean product photo of a single floral dress"
    },
    "shoes": {
        "sneakers": "a clean product photo of a single pair of sneakers",
        "running_shoes": "a clean product photo of a single pair of running shoes",
        "canvas_shoes": "a clean product photo of a single pair of canvas shoes",
        "loafers": "a clean product photo of a single pair of loafers",
        "boots": "a clean product photo of a single pair of boots",
        "ankle_boots": "a clean product photo of a single pair of ankle boots",
        "sandals": "a clean product photo of a single pair of sandals",
        "heels": "a clean product photo of a single pair of high heels",
        "flats": "a clean product photo of a single pair of flats"
    }
}

MAIN_CATEGORY_LABEL_MAP = {
    "headwear": "帽子",
    "upper_body": "上身",
    "lower_body": "下身",
    "dress": "連身裙",
    "shoes": "鞋子"
}

FINE_CATEGORY_LABEL_MAP = {
    "baseball_cap": "棒球帽",
    "bucket_hat": "漁夫帽",
    "beanie": "毛帽",
    "beret": "貝雷帽",
    "sun_hat": "遮陽帽",
    "knit_hat": "針織帽",

    "t_shirt": "T 恤",
    "graphic_t_shirt": "圖案 T 恤",
    "long_sleeve_t_shirt": "長袖 T 恤",
    "polo_shirt": "Polo 衫",
    "shirt": "襯衫",
    "oxford_shirt": "牛津襯衫",
    "blouse": "女式上衣",
    "tank_top": "背心",
    "camisole": "細肩帶背心",
    "hoodie": "連帽上衣",
    "sweatshirt": "大學T",
    "knit_sweater": "針織毛衣",
    "cardigan": "開襟衫",
    "denim_jacket": "牛仔外套",
    "bomber_jacket": "飛行外套",
    "blazer": "西裝外套",
    "coat": "大衣",
    "trench_coat": "風衣",
    "puffer_jacket": "羽絨外套",
    "vest": "背心外套",
    "windbreaker": "防風外套",

    "jeans": "牛仔褲",
    "trousers": "長褲",
    "wide_leg_pants": "寬褲",
    "cargo_pants": "工裝褲",
    "jogger_pants": "束口褲",
    "leggings": "內搭褲",
    "shorts": "短褲",
    "denim_shorts": "牛仔短褲",
    "mini_skirt": "短裙",
    "midi_skirt": "中長裙",
    "pleated_skirt": "百褶裙",
    "denim_skirt": "牛仔裙",

    "mini_dress": "短洋裝",
    "midi_dress": "中長洋裝",
    "maxi_dress": "長洋裝",
    "shirt_dress": "襯衫洋裝",
    "slip_dress": "細肩帶洋裝",
    "knit_dress": "針織洋裝",
    "floral_dress": "碎花洋裝",

    "sneakers": "休閒鞋",
    "running_shoes": "跑鞋",
    "canvas_shoes": "帆布鞋",
    "loafers": "樂福鞋",
    "boots": "靴子",
    "ankle_boots": "短靴",
    "sandals": "涼鞋",
    "heels": "高跟鞋",
    "flats": "平底鞋"
}


def _predict_from_label_map(image, label_map: dict):
    keys = list(label_map.keys())
    prompts = list(label_map.values())

    best_prompt, score = predict_best(image, prompts)
    best_index = prompts.index(best_prompt)
    best_key = keys[best_index]

    return best_key, score


def infer_style(main_category: str, fine_category: str) -> str:
    if fine_category in {"blazer", "shirt", "oxford_shirt", "trench_coat", "loafers", "heels"}:
        return "formal"

    if fine_category in {"running_shoes", "leggings", "jogger_pants", "windbreaker"}:
        return "sport"

    return "casual"


def infer_season(main_category: str, fine_category: str) -> str:
    if fine_category in {
        "tank_top", "camisole", "shorts", "denim_shorts", "sandals", "slip_dress"
    }:
        return "summer"

    if fine_category in {
        "coat", "trench_coat", "puffer_jacket", "knit_sweater", "cardigan",
        "beanie", "knit_hat", "boots", "ankle_boots"
    }:
        return "winter"

    return "spring_autumn"


def classify_category(image):
    # 第一階段：大類
    main_key, main_score = _predict_from_label_map(image, STAGE1_LABELS)

    # 第二階段：細類
    fine_key, fine_score = _predict_from_label_map(image, STAGE2_LABELS[main_key])

    style = infer_style(main_key, fine_key)
    season = infer_season(main_key, fine_key)

    return {
        "mainCategoryKey": main_key,
        "mainCategory": MAIN_CATEGORY_LABEL_MAP[main_key],
        "categoryKey": fine_key,
        "category": FINE_CATEGORY_LABEL_MAP[fine_key],
        "style": style,
        "season": season,
        "scores": {
            "mainCategory": float(main_score),
            "category": float(fine_score)
        },
        "score": float(fine_score)
    }