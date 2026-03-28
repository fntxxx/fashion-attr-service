from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

from fashion_attr_service.models.fashion_siglip_model import (
    encode_image_feature,
    score_texts_with_image_feature,
)
from fashion_attr_service.services.crop_garment import crop_image_by_bbox
from fashion_attr_service.services.detect_garment import detect_main_garment
from fashion_attr_service.services.extract_color import extract_color
from fashion_attr_service.utils.color_tags import COLOR_TONE_TO_UI, COLOR_UI_OPTIONS, COLOR_VALUE_TO_LABEL
from fashion_attr_service.utils.scoring import build_candidates


OCCASION_OPTIONS = [
    ("social", "社交聚會"),
    ("campus_casual", "校園休閒"),
    ("business_casual", "商務休閒"),
    ("professional", "專業職場"),
]

SEASON_OPTIONS = [
    ("spring", "春季"),
    ("summer", "夏季"),
    ("autumn", "秋季"),
    ("winter", "冬季"),
]


# 舊版 rule-based mapping 降級為 prior，不能再直接決策。
OCCASION_PRIORS: dict[str, dict[str, float]] = {
    "t_shirt": {"campus_casual": 0.88, "social": 0.38, "business_casual": 0.18, "professional": 0.08},
    "shirt": {"campus_casual": 0.52, "social": 0.58, "business_casual": 0.83, "professional": 0.72},
    "tank_top": {"campus_casual": 0.70, "social": 0.48, "business_casual": 0.12, "professional": 0.05},
    "hoodie": {"campus_casual": 0.90, "social": 0.26, "business_casual": 0.10, "professional": 0.05},
    "sweatshirt": {"campus_casual": 0.86, "social": 0.30, "business_casual": 0.15, "professional": 0.06},
    "knit_sweater": {"campus_casual": 0.62, "social": 0.54, "business_casual": 0.56, "professional": 0.36},
    "cardigan": {"campus_casual": 0.42, "social": 0.46, "business_casual": 0.72, "professional": 0.50},
    "denim_jacket": {"campus_casual": 0.82, "social": 0.42, "business_casual": 0.18, "professional": 0.08},
    "blazer": {"campus_casual": 0.08, "social": 0.44, "business_casual": 0.88, "professional": 0.92},
    "coat": {"campus_casual": 0.42, "social": 0.50, "business_casual": 0.68, "professional": 0.64},
    "puffer_jacket": {"campus_casual": 0.74, "social": 0.24, "business_casual": 0.12, "professional": 0.08},
    "vest": {"campus_casual": 0.44, "social": 0.34, "business_casual": 0.42, "professional": 0.26},
    "windbreaker": {"campus_casual": 0.72, "social": 0.20, "business_casual": 0.10, "professional": 0.05},
    "jeans": {"campus_casual": 0.88, "social": 0.42, "business_casual": 0.22, "professional": 0.08},
    "trousers": {"campus_casual": 0.22, "social": 0.40, "business_casual": 0.84, "professional": 0.92},
    "wide_leg_pants": {"campus_casual": 0.28, "social": 0.38, "business_casual": 0.72, "professional": 0.64},
    "leggings": {"campus_casual": 0.72, "social": 0.18, "business_casual": 0.06, "professional": 0.04},
    "shorts": {"campus_casual": 0.82, "social": 0.32, "business_casual": 0.08, "professional": 0.04},
    "mini_skirt": {"campus_casual": 0.58, "social": 0.70, "business_casual": 0.22, "professional": 0.10},
    "midi_skirt": {"campus_casual": 0.42, "social": 0.66, "business_casual": 0.56, "professional": 0.28},
    "mini_dress": {"campus_casual": 0.36, "social": 0.84, "business_casual": 0.24, "professional": 0.10},
    "midi_dress": {"campus_casual": 0.28, "social": 0.78, "business_casual": 0.48, "professional": 0.18},
    "sneakers": {"campus_casual": 0.90, "social": 0.26, "business_casual": 0.12, "professional": 0.04},
    "boots": {"campus_casual": 0.34, "social": 0.44, "business_casual": 0.20, "professional": 0.08},
    "sandals": {"campus_casual": 0.58, "social": 0.52, "business_casual": 0.08, "professional": 0.04},
    "heels": {"campus_casual": 0.12, "social": 0.76, "business_casual": 0.42, "professional": 0.22},
    "flats": {"campus_casual": 0.26, "social": 0.50, "business_casual": 0.46, "professional": 0.18},
    "bucket_hat": {"campus_casual": 0.72, "social": 0.24, "business_casual": 0.06, "professional": 0.02},
    "beanie": {"campus_casual": 0.66, "social": 0.16, "business_casual": 0.04, "professional": 0.02},
    "hat": {"campus_casual": 0.52, "social": 0.30, "business_casual": 0.08, "professional": 0.04},
}

SEASON_PRIORS: dict[str, dict[str, float]] = {
    "tank_top": {"spring": 0.36, "summer": 0.92, "autumn": 0.14, "winter": 0.04},
    "t_shirt": {"spring": 0.70, "summer": 0.76, "autumn": 0.48, "winter": 0.10},
    "shirt": {"spring": 0.72, "summer": 0.40, "autumn": 0.66, "winter": 0.18},
    "hoodie": {"spring": 0.44, "summer": 0.06, "autumn": 0.82, "winter": 0.70},
    "sweatshirt": {"spring": 0.42, "summer": 0.04, "autumn": 0.84, "winter": 0.72},
    "knit_sweater": {"spring": 0.22, "summer": 0.02, "autumn": 0.74, "winter": 0.92},
    "cardigan": {"spring": 0.80, "summer": 0.08, "autumn": 0.62, "winter": 0.58},
    "denim_jacket": {"spring": 0.54, "summer": 0.06, "autumn": 0.80, "winter": 0.56},
    "blazer": {"spring": 0.58, "summer": 0.18, "autumn": 0.72, "winter": 0.44},
    "coat": {"spring": 0.10, "summer": 0.01, "autumn": 0.48, "winter": 0.96},
    "puffer_jacket": {"spring": 0.04, "summer": 0.01, "autumn": 0.28, "winter": 0.99},
    "vest": {"spring": 0.42, "summer": 0.26, "autumn": 0.66, "winter": 0.34},
    "windbreaker": {"spring": 0.62, "summer": 0.16, "autumn": 0.74, "winter": 0.32},
    "jeans": {"spring": 0.64, "summer": 0.18, "autumn": 0.74, "winter": 0.44},
    "trousers": {"spring": 0.62, "summer": 0.22, "autumn": 0.74, "winter": 0.40},
    "wide_leg_pants": {"spring": 0.58, "summer": 0.30, "autumn": 0.70, "winter": 0.34},
    "leggings": {"spring": 0.26, "summer": 0.02, "autumn": 0.70, "winter": 0.88},
    "shorts": {"spring": 0.34, "summer": 0.96, "autumn": 0.18, "winter": 0.02},
    "mini_skirt": {"spring": 0.72, "summer": 0.66, "autumn": 0.38, "winter": 0.08},
    "midi_skirt": {"spring": 0.58, "summer": 0.42, "autumn": 0.68, "winter": 0.24},
    "mini_dress": {"spring": 0.60, "summer": 0.82, "autumn": 0.32, "winter": 0.06},
    "midi_dress": {"spring": 0.56, "summer": 0.58, "autumn": 0.56, "winter": 0.16},
    "sneakers": {"spring": 0.66, "summer": 0.46, "autumn": 0.68, "winter": 0.26},
    "boots": {"spring": 0.18, "summer": 0.02, "autumn": 0.68, "winter": 0.94},
    "sandals": {"spring": 0.28, "summer": 0.96, "autumn": 0.08, "winter": 0.01},
    "heels": {"spring": 0.42, "summer": 0.52, "autumn": 0.42, "winter": 0.12},
    "flats": {"spring": 0.62, "summer": 0.46, "autumn": 0.40, "winter": 0.12},
    "bucket_hat": {"spring": 0.48, "summer": 0.64, "autumn": 0.20, "winter": 0.06},
    "beanie": {"spring": 0.12, "summer": 0.01, "autumn": 0.46, "winter": 0.92},
    "hat": {"spring": 0.44, "summer": 0.62, "autumn": 0.24, "winter": 0.08},
}

DEFAULT_OCCASION_PRIOR = {"social": 0.28, "campus_casual": 0.32, "business_casual": 0.26, "professional": 0.20}
DEFAULT_SEASON_PRIOR = {"spring": 0.55, "summer": 0.20, "autumn": 0.58, "winter": 0.30}


OCCASION_PROMPTS: dict[str, list[str]] = {
    "social": [
        "a clean product photo of a fashion item suitable for social gatherings, dates, dinners, or going out",
        "a stylish garment commonly worn for social occasions and weekend outings",
    ],
    "campus_casual": [
        "a clean product photo of a casual everyday clothing item for campus, commuting, and relaxed daily wear",
        "a practical and comfortable casual fashion item for daily outfits and student style",
    ],
    "business_casual": [
        "a clean product photo of a fashion item suitable for business casual outfits, office smart casual, and polished daily workwear",
        "a neat clothing item that fits smart casual office wear without being fully formal",
    ],
    "professional": [
        "a clean product photo of a fashion item suitable for professional office wear, formal meetings, and polished work outfits",
        "a refined structured garment for professional workplace styling and formal business presentation",
    ],
}

SEASON_PROMPTS: dict[str, list[str]] = {
    "spring": [
        "a clean product photo of a clothing item suitable for mild spring weather, light layering, and transitional temperatures",
        "a garment appropriate for spring with moderate coverage and breathable comfort",
    ],
    "summer": [
        "a clean product photo of a lightweight breathable clothing item suitable for hot summer weather",
        "a garment for warm summer outfits with airy fabric, lighter coverage, or cooling wear",
    ],
    "autumn": [
        "a clean product photo of a clothing item suitable for cool autumn weather and light-to-medium layering",
        "a garment appropriate for fall outfits with slightly warmer coverage and transitional layering",
    ],
    "winter": [
        "a clean product photo of a warm clothing item suitable for cold winter weather, insulation, and heavier coverage",
        "a garment for winter outfits with warmth, thicker material, or cold-weather protection",
    ],
}

COLOR_PROMPTS: dict[str, list[str]] = {
    "light_beige": [
        "a clean product photo of a single garment whose dominant clothing color is light beige, cream, ivory, off-white, or soft white",
        "a fashion item with a mainly light beige or creamy neutral tone",
    ],
    "dark_gray_black": [
        "a clean product photo of a single garment whose dominant clothing color is black, charcoal, or very dark gray",
        "a fashion item with a mainly black or deep dark neutral tone",
    ],
    "neutral_gray": [
        "a clean product photo of a single garment whose dominant clothing color is gray or cool neutral gray",
        "a fashion item with a mainly medium gray neutral tone",
    ],
    "earth_brown": [
        "a clean product photo of a single garment whose dominant clothing color is brown, khaki, camel, tan, or earthy brown",
        "a fashion item with a mainly earth brown, khaki, or camel tone",
    ],
    "butter_yellow": [
        "a clean product photo of a single garment whose dominant clothing color is yellow, butter yellow, or warm pastel yellow",
        "a fashion item with a mainly yellow tone",
    ],
    "warm_orange_red": [
        "a clean product photo of a single garment whose dominant clothing color is red, orange red, coral, or warm red",
        "a fashion item with a mainly warm red or orange-red tone",
    ],
    "rose_pink": [
        "a clean product photo of a single garment whose dominant clothing color is pink, rose pink, blush pink, or dusty pink",
        "a fashion item with a mainly pink or rose tone",
    ],
    "natural_green": [
        "a clean product photo of a single garment whose dominant clothing color is green, olive, sage, or natural green",
        "a fashion item with a mainly green tone",
    ],
    "fresh_blue": [
        "a clean product photo of a single garment whose dominant clothing color is blue, navy, denim blue, or fresh blue",
        "a fashion item with a mainly blue tone",
    ],
    "elegant_purple": [
        "a clean product photo of a single garment whose dominant clothing color is purple, lavender, lilac, or plum",
        "a fashion item with a mainly purple tone",
    ],
}


@dataclass(frozen=True)
class AttributeSelectionConfig:
    min_score: float
    relative_ratio: float
    max_selected: int
    min_margin_from_prior: float = 0.0


OCCASION_SELECTION = AttributeSelectionConfig(
    min_score=0.19,
    relative_ratio=0.58,
    max_selected=3,
)

SEASON_SELECTION = AttributeSelectionConfig(
    min_score=0.16,
    relative_ratio=0.50,
    max_selected=3,
)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_prior_map(
    fine_category: str,
    default_prior: Mapping[str, float],
    category_prior_map: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    return dict(category_prior_map.get(fine_category, default_prior))


def _score_prompt_ensemble(
    image_features,
    prompt_map: Mapping[str, Sequence[str]],
    label_map: Mapping[str, str],
    *,
    prior_map: Mapping[str, float] | None = None,
    prior_bias_strength: float,
    candidate_temperature: float | None = None,
) -> dict[str, Any]:
    raw_prompt_score_map: dict[str, float] = {}
    prompt_breakdown: dict[str, list[dict[str, float | str]]] = {}

    for value, prompts in prompt_map.items():
        prompt_results = score_texts_with_image_feature(image_features, list(prompts))
        prompt_scores = [float(item["score"]) for item in prompt_results]
        raw_prompt_score_map[value] = _mean(prompt_scores)
        prompt_breakdown[value] = [
            {"prompt": str(item["label"]), "score": float(item["score"])}
            for item in prompt_results
        ]

    combined_score_map: dict[str, float] = {}
    prior_component_map: dict[str, float] = {}
    for value, raw_score in raw_prompt_score_map.items():
        prior_value = float((prior_map or {}).get(value, 0.5))
        prior_component = (prior_value - 0.5) * prior_bias_strength
        prior_component_map[value] = float(prior_component)
        combined_score_map[value] = float(raw_score + prior_component)

    candidates, normalized_map = build_candidates(
        combined_score_map,
        label_map,
        temperature=candidate_temperature,
    )

    return {
        "candidates": candidates,
        "candidateScoreMap": normalized_map,
        "rawPromptScoreMap": raw_prompt_score_map,
        "priorMap": dict(prior_map or {}),
        "priorComponentMap": prior_component_map,
        "combinedScoreMap": combined_score_map,
        "promptBreakdown": prompt_breakdown,
    }


def _select_multi_label(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: AttributeSelectionConfig,
) -> dict[str, Any]:
    if not candidates:
        return {
            "selected": [],
            "decision": "empty",
            "top_score": 0.0,
            "selected_scores": {},
            "threshold": config.min_score,
            "relative_ratio": config.relative_ratio,
            "max_selected": config.max_selected,
        }

    top_score = float(candidates[0]["score"])
    relative_floor = top_score * config.relative_ratio
    effective_floor = max(config.min_score, relative_floor)

    selected = [
        str(item["value"])
        for item in candidates
        if float(item["score"]) >= effective_floor
    ][: config.max_selected]

    if not selected:
        selected = [str(candidates[0]["value"])]

    selected_scores = {
        str(item["value"]): float(item["score"])
        for item in candidates
        if str(item["value"]) in selected
    }

    return {
        "selected": selected,
        "decision": "threshold_multi" if len(selected) > 1 else "threshold_single",
        "top_score": top_score,
        "selected_scores": selected_scores,
        "threshold": config.min_score,
        "relative_ratio": config.relative_ratio,
        "effective_floor": effective_floor,
        "max_selected": config.max_selected,
    }


def infer_occasions(image_features, main_category: str, fine_category: str) -> dict[str, Any]:
    del main_category
    label_map = {value: label for value, label in OCCASION_OPTIONS}
    prior_map = _build_prior_map(fine_category, DEFAULT_OCCASION_PRIOR, OCCASION_PRIORS)

    scored = _score_prompt_ensemble(
        image_features,
        OCCASION_PROMPTS,
        label_map,
        prior_map=prior_map,
        prior_bias_strength=0.22,
        candidate_temperature=0.06,
    )
    selection = _select_multi_label(scored["candidates"], config=OCCASION_SELECTION)

    selected = selection["selected"]
    legacy_style = "casual"
    if "professional" in selected:
        legacy_style = "formal"
    elif fine_category in {"hoodie", "leggings", "windbreaker", "sneakers"}:
        legacy_style = "sport"
    elif "business_casual" in selected:
        legacy_style = "smart_casual"

    return {
        "selected": selected,
        "candidates": scored["candidates"],
        "candidateScoreMap": scored["candidateScoreMap"],
        "selectionDebug": selection,
        "rawPromptScoreMap": scored["rawPromptScoreMap"],
        "priorMap": scored["priorMap"],
        "priorComponentMap": scored["priorComponentMap"],
        "combinedScoreMap": scored["combinedScoreMap"],
        "promptBreakdown": scored["promptBreakdown"],
        "legacy_style": legacy_style,
    }


def infer_seasons(image_features, main_category: str, fine_category: str) -> dict[str, Any]:
    del main_category
    label_map = {value: label for value, label in SEASON_OPTIONS}
    prior_map = _build_prior_map(fine_category, DEFAULT_SEASON_PRIOR, SEASON_PRIORS)

    scored = _score_prompt_ensemble(
        image_features,
        SEASON_PROMPTS,
        label_map,
        prior_map=prior_map,
        prior_bias_strength=0.18,
        candidate_temperature=0.055,
    )
    selection = _select_multi_label(scored["candidates"], config=SEASON_SELECTION)

    return {
        "selected": selection["selected"],
        "candidates": scored["candidates"],
        "candidateScoreMap": scored["candidateScoreMap"],
        "selectionDebug": selection,
        "rawPromptScoreMap": scored["rawPromptScoreMap"],
        "priorMap": scored["priorMap"],
        "priorComponentMap": scored["priorComponentMap"],
        "combinedScoreMap": scored["combinedScoreMap"],
        "promptBreakdown": scored["promptBreakdown"],
    }


def _build_color_prior_map(heuristic_color_value: str | None) -> dict[str, float]:
    selected = heuristic_color_value
    score_map: dict[str, float] = {value: 0.20 for value, _label in COLOR_UI_OPTIONS}

    if selected in score_map:
        score_map[selected] = 0.86

    if selected == "light_beige":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.34)
        score_map["earth_brown"] = max(score_map["earth_brown"], 0.30)
    elif selected == "dark_gray_black":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.42)
    elif selected == "neutral_gray":
        score_map["dark_gray_black"] = max(score_map["dark_gray_black"], 0.40)
        score_map["light_beige"] = max(score_map["light_beige"], 0.24)
    elif selected == "earth_brown":
        score_map["light_beige"] = max(score_map["light_beige"], 0.36)
        score_map["butter_yellow"] = max(score_map["butter_yellow"], 0.28)
    elif selected == "butter_yellow":
        score_map["earth_brown"] = max(score_map["earth_brown"], 0.30)
        score_map["light_beige"] = max(score_map["light_beige"], 0.28)
    elif selected == "warm_orange_red":
        score_map["rose_pink"] = max(score_map["rose_pink"], 0.30)
    elif selected == "rose_pink":
        score_map["warm_orange_red"] = max(score_map["warm_orange_red"], 0.26)
    elif selected == "fresh_blue":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.22)
    elif selected == "elegant_purple":
        score_map["rose_pink"] = max(score_map["rose_pink"], 0.22)

    return score_map


def prepare_color_focus_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    detection = detect_main_garment(image)
    if detection.get("detected") and detection.get("bbox"):
        focused = crop_image_by_bbox(image, detection["bbox"], padding_ratio=0.04)
        source = "detected_bbox"
    else:
        focused = image
        source = "full_image_fallback"

    return focused, {
        "source": source,
        "detection": detection,
        "focused_size": list(focused.size),
    }


def infer_color(image: Image.Image, *, model_backend: str | None = None) -> dict[str, Any]:
    focused_image, focus_debug = prepare_color_focus_image(image)
    image_features = encode_image_feature(focused_image, model_backend=model_backend)

    heuristic_tone = extract_color(focused_image)
    heuristic_color_value = COLOR_TONE_TO_UI.get(heuristic_tone)
    prior_map = _build_color_prior_map(heuristic_color_value)
    label_map = {value: label for value, label in COLOR_UI_OPTIONS}

    scored = _score_prompt_ensemble(
        image_features,
        COLOR_PROMPTS,
        label_map,
        prior_map=prior_map,
        prior_bias_strength=0.16,
        candidate_temperature=0.045,
    )
    candidates = scored["candidates"]
    color_value = str(candidates[0]["value"]) if candidates else (heuristic_color_value or "neutral_gray")
    color_label = COLOR_VALUE_TO_LABEL.get(color_value, "中性灰")

    return {
        "color": color_value,
        "colorLabel": color_label,
        "candidates": candidates,
        "scoreMap": scored["candidateScoreMap"],
        "candidateScoreMap": scored["candidateScoreMap"],
        "rawPromptScoreMap": scored["rawPromptScoreMap"],
        "priorMap": scored["priorMap"],
        "priorComponentMap": scored["priorComponentMap"],
        "combinedScoreMap": scored["combinedScoreMap"],
        "promptBreakdown": scored["promptBreakdown"],
        "heuristicTone": heuristic_tone,
        "heuristicColorValue": heuristic_color_value,
        "focusDebug": focus_debug,
    }