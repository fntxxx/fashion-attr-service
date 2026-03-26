from __future__ import annotations

CATEGORY_UI_OPTIONS = [
    ("top", "上衣"),
    ("pants", "褲子"),
    ("skirt", "裙子"),
    ("dress", "連身裙"),
    ("outer", "外套"),
    ("shoes", "鞋子"),
]

CATEGORY_VALUE_TO_LABEL = {value: label for value, label in CATEGORY_UI_OPTIONS}

MAIN_CATEGORY_TO_UI = {
    "upper_body": "top",
    "pants": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "shoes": "shoes",
    # 目前 UI 沒有帽子，先保守 fallback 到 top
    "headwear": "top",
}

OUTER_FINE_KEYS = {
    "denim_jacket",
    "blazer",
    "coat",
    "puffer_jacket",
    "vest",
    "windbreaker",
    "cardigan",
}
