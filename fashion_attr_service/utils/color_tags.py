from __future__ import annotations

from fashion_attr_service.utils.scoring import build_candidates


COLOR_UI_OPTIONS = [
    ("light_beige", "淺米白"),
    ("dark_gray_black", "深灰黑"),
    ("neutral_gray", "中性灰"),
    ("earth_brown", "大地棕"),
    ("butter_yellow", "奶油黃"),
    ("warm_orange_red", "暖橘紅"),
    ("rose_pink", "粉桃紅"),
    ("natural_green", "自然綠"),
    ("fresh_blue", "清爽藍"),
    ("elegant_purple", "優雅紫"),
]

COLOR_TONE_TO_UI = {
    "白色系": "light_beige",
    "米色系": "light_beige",
    "黑色系": "dark_gray_black",
    "灰色系": "neutral_gray",
    "卡其色系": "earth_brown",
    "咖啡色系": "earth_brown",
    "黃色系": "butter_yellow",
    "紅色系": "warm_orange_red",
    "粉紅色系": "rose_pink",
    "綠色系": "natural_green",
    "藍色系": "fresh_blue",
    "紫色系": "elegant_purple",
}

COLOR_VALUE_TO_LABEL = {value: label for value, label in COLOR_UI_OPTIONS}


def _build_color_score_map(color_tone: str) -> dict[str, float]:
    selected_value = COLOR_TONE_TO_UI.get(color_tone)
    score_map: dict[str, float] = {}

    for value, _label in COLOR_UI_OPTIONS:
        score = 0.04

        if value == selected_value:
            score = 0.92

        if color_tone in {"白色系", "米色系"}:
            if value == "neutral_gray":
                score = max(score, 0.18)
            if value == "earth_brown":
                score = max(score, 0.12)

        if color_tone in {"黑色系", "灰色系"}:
            if value in {"dark_gray_black", "neutral_gray"}:
                score = max(score, 0.72 if value != selected_value else score)

        if color_tone in {"卡其色系", "咖啡色系", "黃色系"}:
            if value == "light_beige":
                score = max(score, 0.24)
            if color_tone == "黃色系" and value == "earth_brown":
                score = max(score, 0.16)

        if color_tone == "紅色系" and value == "rose_pink":
            score = max(score, 0.26)

        if color_tone == "粉紅色系" and value == "warm_orange_red":
            score = max(score, 0.18)

        score_map[value] = float(score)

    return score_map


def build_color_payload(color_tone: str):
    label_map = {value: label for value, label in COLOR_UI_OPTIONS}
    score_map = _build_color_score_map(color_tone)
    candidates, normalized_map = build_candidates(score_map, label_map)

    color_value = COLOR_TONE_TO_UI.get(color_tone)
    if not color_value and candidates:
        color_value = candidates[0]["value"]

    color_label = COLOR_VALUE_TO_LABEL.get(color_value, "中性灰")

    return {
        "color": color_value,
        "colorLabel": color_label,
        "candidates": candidates,
        "scoreMap": normalized_map,
    }