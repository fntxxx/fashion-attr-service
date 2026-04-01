from __future__ import annotations

from typing import Final

MAIN_CATEGORY_LABEL_MAP: Final[dict[str, str]] = {
    "headwear": "帽子",
    "upper_body": "上身",
    "pants": "褲子",
    "skirt": "裙子",
    "dress": "連身裙",
    "shoes": "鞋子",
}

HEADWEAR_CATEGORY_MAP: Final[dict[str, str]] = {
    "bucket_hat": "漁夫帽",
    "beanie": "毛帽",
    "hat": "帽子",
}

UPPER_BODY_CATEGORY_MAP: Final[dict[str, str]] = {
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

PANTS_CATEGORY_MAP: Final[dict[str, str]] = {
    "jeans": "牛仔褲",
    "trousers": "長褲",
    "wide_leg_pants": "寬褲",
    "leggings": "內搭褲",
    "shorts": "短褲",
}

SKIRT_CATEGORY_MAP: Final[dict[str, str]] = {
    "mini_skirt": "短裙",
    "midi_skirt": "中長裙",
}

DRESS_CATEGORY_MAP: Final[dict[str, str]] = {
    "mini_dress": "短洋裝",
    "midi_dress": "中長洋裝",
}

SHOES_CATEGORY_MAP: Final[dict[str, str]] = {
    "sneakers": "休閒鞋",
    "boots": "靴子",
    "sandals": "涼鞋",
    "heels": "高跟鞋",
    "flats": "平底鞋",
}

FINE_CATEGORY_MAPS: Final[dict[str, dict[str, str]]] = {
    "headwear": HEADWEAR_CATEGORY_MAP,
    "upper_body": UPPER_BODY_CATEGORY_MAP,
    "pants": PANTS_CATEGORY_MAP,
    "skirt": SKIRT_CATEGORY_MAP,
    "dress": DRESS_CATEGORY_MAP,
    "shoes": SHOES_CATEGORY_MAP,
}

FINE_CATEGORY_DEFAULTS: Final[dict[str, str]] = {
    "headwear": "hat",
    "upper_body": "shirt",
    "pants": "trousers",
    "skirt": "midi_skirt",
    "dress": "midi_dress",
    "shoes": "sneakers",
}

TOP_FINE_KEYS: Final[set[str]] = {
    "t_shirt",
    "shirt",
    "tank_top",
    "hoodie",
    "sweatshirt",
    "knit_sweater",
}

OUTER_FINE_KEYS: Final[set[str]] = {
    "blazer",
    "coat",
    "puffer_jacket",
    "vest",
    "windbreaker",
    "denim_jacket",
    "cardigan",
}


def get_main_category_label(main_key: str) -> str:
    return MAIN_CATEGORY_LABEL_MAP[main_key]



def get_fine_category_map(main_key: str) -> dict[str, str]:
    return FINE_CATEGORY_MAPS[main_key]



def get_fine_category_label(main_key: str, category_key: str) -> str:
    return get_fine_category_map(main_key)[category_key]



def normalize_category_key(main_key: str, category_key: str) -> str:
    fine_category_map = get_fine_category_map(main_key)
    if category_key in fine_category_map:
        return category_key
    return FINE_CATEGORY_DEFAULTS[main_key]
