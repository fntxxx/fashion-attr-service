from __future__ import annotations

from typing import Any


COLOR_UI_OPTIONS = [
    ("light_beige", "淺米白"),
    ("dark_gray_black", "深灰黑"),
    ("neutral_gray", "中性灰"),
    ("earth_brown", "大地棕"),
    ("warm_orange_red", "暖橘紅"),
    ("rose_pink", "粉嫩玫瑰"),
    ("natural_green", "自然綠"),
    ("fresh_blue", "清爽藍"),
    ("elegant_purple", "優雅紫"),
    ("pattern", "花紋圖案"),
]


COLOR_TONE_TO_UI = {
    "白色系": ["light_beige"],
    "米色系": ["light_beige"],
    "黑色系": ["dark_gray_black"],
    "灰色系": ["neutral_gray"],
    "卡其色系": ["earth_brown"],
    "咖啡色系": ["earth_brown"],
    "紅色系": ["warm_orange_red"],
    "綠色系": ["natural_green"],
    "藍色系": ["fresh_blue"],
    "紫色系": ["elegant_purple"],
    "花紋圖案": ["pattern"],
}

COLOR_VALUE_TO_LABEL = {value: label for value, label in COLOR_UI_OPTIONS}


def color_tone_to_tags(color_tone: str):
    values = COLOR_TONE_TO_UI.get(color_tone, [])
    return [COLOR_VALUE_TO_LABEL[value] for value in values]


def _build_color_candidates(color_tone: str) -> list[dict[str, Any]]:
    selected_values = set(COLOR_TONE_TO_UI.get(color_tone, []))
    candidates = []

    for value, label in COLOR_UI_OPTIONS:
        score = 0.04

        if value in selected_values:
            score = 0.92

        if color_tone in {"白色系", "米色系"}:
            if value == "neutral_gray":
                score = max(score, 0.18)
            if value == "earth_brown":
                score = max(score, 0.12)

        if color_tone in {"黑色系", "灰色系"}:
            if value in {"dark_gray_black", "neutral_gray"}:
                score = max(score, 0.72 if value not in selected_values else score)

        if color_tone in {"卡其色系", "咖啡色系"}:
            if value == "light_beige":
                score = max(score, 0.24)

        if color_tone == "紅色系":
            if value == "rose_pink":
                score = max(score, 0.26)

        candidates.append({
            "value": value,
            "label": label,
            "score": float(score),
        })

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def build_color_payload(color_tone: str):
    selected = COLOR_TONE_TO_UI.get(color_tone, [])
    candidates = _build_color_candidates(color_tone)

    if not selected and candidates:
        selected = [candidates[0]["value"]]

    return {
        "colorTone": color_tone,
        "colorTags": color_tone_to_tags(color_tone),
        "colors": {
            "selected": selected[:2],
            "candidates": candidates,
            "threshold": 0.58,
            "maxSelected": 2,
        },
    }