from __future__ import annotations

from fashion_attr_service.utils.category_catalog import OUTER_FINE_KEYS

SERVICE_NAME = "fashion-attr-service"
API_VERSION = "1.3.0"

CATEGORY_UI_OPTIONS = [
    ("top", "上衣"),
    ("bottom", "褲子"),
    ("outerwear", "外套"),
    ("shoes", "鞋子"),
    ("skirt", "裙子"),
    ("dress", "連身裙"),
]

CATEGORY_VALUE_TO_LABEL = {value: label for value, label in CATEGORY_UI_OPTIONS}

MAIN_CATEGORY_TO_UI = {
    "upper_body": "top",
    "pants": "bottom",
    "skirt": "skirt",
    "dress": "dress",
    "shoes": "shoes",
    # 目前 UI 沒有帽子，先保守 fallback 到 top
    "headwear": "top",
}

PUBLIC_OCCASION_VALUE_MAP = {
    "social": "socialGathering",
    "campus_casual": "campusCasual",
    "business_casual": "businessCasual",
    "professional": "professional",
}

PUBLIC_SEASON_VALUE_MAP = {
    "spring": "spring",
    "summer": "summer",
    "autumn": "autumn",
    "winter": "winter",
}

PUBLIC_COLOR_VALUE_MAP = {
    "light_beige": "white",
    "dark_gray_black": "black",
    "neutral_gray": "gray",
    "earth_brown": "brown",
    "butter_yellow": "yellow",
    "warm_orange_red": "orange",
    "rose_pink": "pink",
    "natural_green": "green",
    "fresh_blue": "blue",
    "elegant_purple": "purple",
}



ERROR_CODE_PREDICT_REJECTED = "predict_rejected"
ERROR_CODE_REQUEST_VALIDATION = "request_validation_error"
ERROR_CODE_HTTP_EXCEPTION = "http_exception"
ERROR_CODE_INTERNAL_SERVER = "internal_server_error"
ERROR_CODE_WARMUP_FAILED = "warmup_failed"

ERROR_MESSAGE_PREDICT_REJECTED = "輸入圖片未通過服飾商品圖驗證。"
ERROR_MESSAGE_REQUEST_VALIDATION = "請求參數驗證失敗。"
ERROR_MESSAGE_INTERNAL_SERVER = "伺服器發生未預期錯誤。"
ERROR_MESSAGE_WARMUP_FAILED = "模型 warmup 失敗。"

ERROR_CODE_UNAUTHORIZED = "unauthorized"
ERROR_MESSAGE_UNAUTHORIZED = "缺少或無效的 API Token。"
