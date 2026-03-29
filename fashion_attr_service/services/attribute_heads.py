from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
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
    "shirt": {"spring": 0.76, "summer": 0.52, "autumn": 0.56, "winter": 0.18},
    "hoodie": {"spring": 0.44, "summer": 0.06, "autumn": 0.82, "winter": 0.70},
    "sweatshirt": {"spring": 0.42, "summer": 0.04, "autumn": 0.84, "winter": 0.72},
    "knit_sweater": {"spring": 0.18, "summer": 0.02, "autumn": 0.66, "winter": 0.94},
    "cardigan": {"spring": 0.82, "summer": 0.08, "autumn": 0.52, "winter": 0.60},
    "denim_jacket": {"spring": 0.54, "summer": 0.06, "autumn": 0.80, "winter": 0.56},
    "blazer": {"spring": 0.58, "summer": 0.18, "autumn": 0.72, "winter": 0.44},
    "coat": {"spring": 0.10, "summer": 0.01, "autumn": 0.48, "winter": 0.96},
    "puffer_jacket": {"spring": 0.04, "summer": 0.01, "autumn": 0.28, "winter": 0.99},
    "vest": {"spring": 0.42, "summer": 0.26, "autumn": 0.66, "winter": 0.34},
    "windbreaker": {"spring": 0.62, "summer": 0.16, "autumn": 0.74, "winter": 0.32},
    "jeans": {"spring": 0.54, "summer": 0.20, "autumn": 0.76, "winter": 0.50},
    "trousers": {"spring": 0.58, "summer": 0.28, "autumn": 0.76, "winter": 0.48},
    "wide_leg_pants": {"spring": 0.58, "summer": 0.30, "autumn": 0.70, "winter": 0.34},
    "leggings": {"spring": 0.26, "summer": 0.02, "autumn": 0.70, "winter": 0.88},
    "shorts": {"spring": 0.34, "summer": 0.96, "autumn": 0.18, "winter": 0.02},
    "mini_skirt": {"spring": 0.72, "summer": 0.66, "autumn": 0.38, "winter": 0.08},
    "midi_skirt": {"spring": 0.62, "summer": 0.50, "autumn": 0.58, "winter": 0.22},
    "mini_dress": {"spring": 0.60, "summer": 0.82, "autumn": 0.32, "winter": 0.06},
    "midi_dress": {"spring": 0.56, "summer": 0.58, "autumn": 0.56, "winter": 0.16},
    "sneakers": {"spring": 0.66, "summer": 0.46, "autumn": 0.68, "winter": 0.26},
    "boots": {"spring": 0.18, "summer": 0.02, "autumn": 0.74, "winter": 0.84},
    "sandals": {"spring": 0.28, "summer": 0.96, "autumn": 0.08, "winter": 0.01},
    "heels": {"spring": 0.42, "summer": 0.52, "autumn": 0.42, "winter": 0.12},
    "flats": {"spring": 0.64, "summer": 0.40, "autumn": 0.34, "winter": 0.12},
    "bucket_hat": {"spring": 0.48, "summer": 0.64, "autumn": 0.20, "winter": 0.06},
    "beanie": {"spring": 0.12, "summer": 0.01, "autumn": 0.46, "winter": 0.92},
    "hat": {"spring": 0.44, "summer": 0.62, "autumn": 0.24, "winter": 0.08},
}

DEFAULT_OCCASION_PRIOR = {"social": 0.28, "campus_casual": 0.32, "business_casual": 0.26, "professional": 0.20}
DEFAULT_SEASON_PRIOR = {"spring": 0.55, "summer": 0.20, "autumn": 0.58, "winter": 0.30}

MAIN_CATEGORY_OCCASION_PRIORS: dict[str, dict[str, float]] = {
    "upper_body": {"social": 0.38, "campus_casual": 0.42, "business_casual": 0.44, "professional": 0.30},
    "pants": {"social": 0.28, "campus_casual": 0.36, "business_casual": 0.54, "professional": 0.46},
    "skirt": {"social": 0.54, "campus_casual": 0.34, "business_casual": 0.38, "professional": 0.22},
    "dress": {"social": 0.66, "campus_casual": 0.26, "business_casual": 0.38, "professional": 0.18},
    "shoes": {"social": 0.40, "campus_casual": 0.48, "business_casual": 0.34, "professional": 0.18},
    "headwear": {"social": 0.22, "campus_casual": 0.62, "business_casual": 0.12, "professional": 0.06},
}

OCCASION_FINE_CATEGORY_PRIOR_STRENGTH: dict[str, float] = {
    "shirt": 0.15,
    "shorts": 0.08,
    "sneakers": 0.08,
}


OCCASION_PROMPTS: dict[str, list[str]] = {
    "social": [
        "a clean product photo of a fashion item suitable for social gatherings, dates, dinners, or going out",
        "a stylish garment commonly worn for social occasions and weekend outings",
        "a fashion item that reads more dressy, outing-ready, or socially styled than purely office-focused",
    ],
    "campus_casual": [
        "a clean product photo of a casual everyday clothing item for campus, commuting, and relaxed daily wear",
        "a practical and comfortable casual fashion item for daily outfits and student style",
        "a laid-back casual clothing item for routine wear, easy styling, and non-formal outfits",
    ],
    "business_casual": [
        "a clean product photo of a fashion item suitable for business casual outfits, office smart casual, and polished daily workwear",
        "a neat clothing item that fits smart casual office wear without being fully formal",
        "a polished but approachable garment for office-ready smart casual styling",
    ],
    "professional": [
        "a clean product photo of a fashion item suitable for professional office wear, formal meetings, and polished work outfits",
        "a refined structured garment for professional workplace styling and formal business presentation",
        "a more formal and professional garment with structured, office-focused presentation",
    ],
}

SEASON_PROMPTS: dict[str, list[str]] = {
    "spring": [
        "a clean product photo of a clothing item suitable for mild spring weather, light layering, and transitional temperatures",
        "a garment appropriate for spring with moderate coverage and breathable comfort",
        "a fashion item for mild weather and lighter transitional outfits rather than peak summer heat or deep winter cold",
    ],
    "summer": [
        "a clean product photo of a lightweight breathable clothing item suitable for hot summer weather",
        "a garment for warm summer outfits with airy fabric, lighter coverage, or cooling wear",
        "a fashion item that feels clearly light, breezy, or hot-weather oriented rather than layered or insulated",
    ],
    "autumn": [
        "a clean product photo of a clothing item suitable for cool autumn weather and light-to-medium layering",
        "a garment appropriate for fall outfits with slightly warmer coverage and transitional layering",
        "a fashion item suited to cooler transitional weather with more warmth than spring or summer",
    ],
    "winter": [
        "a clean product photo of a warm clothing item suitable for cold winter weather, insulation, and heavier coverage",
        "a garment for winter outfits with warmth, thicker material, or cold-weather protection",
        "a fashion item with obvious cold-weather warmth, insulation, or heavier protective coverage",
    ],
}

COLOR_PROMPTS: dict[str, list[str]] = {
    "light_beige": [
        "a clean product photo of a single garment whose dominant clothing color is light beige, cream, ivory, off-white, or soft white",
        "a fashion item with a mainly light beige or creamy neutral tone",
        "a clothing item that appears warm light neutral, off-white, creamy, or beige rather than gray or pure yellow",
    ],
    "dark_gray_black": [
        "a clean product photo of a single garment whose dominant clothing color is black, charcoal, or very dark gray",
        "a fashion item with a mainly black or deep dark neutral tone",
        "a clothing item that reads predominantly black, charcoal, or near-black rather than medium gray",
    ],
    "neutral_gray": [
        "a clean product photo of a single garment whose dominant clothing color is gray or cool neutral gray",
        "a fashion item with a mainly medium gray neutral tone",
        "a clothing item that looks gray, silver gray, or cool neutral rather than beige, white, or black",
    ],
    "earth_brown": [
        "a clean product photo of a single garment whose dominant clothing color is brown, khaki, camel, tan, or earthy brown",
        "a fashion item with a mainly earth brown, khaki, or camel tone",
        "a clothing item that reads more brown or camel than yellow, red, or light beige",
    ],
    "butter_yellow": [
        "a clean product photo of a single garment whose dominant clothing color is yellow, butter yellow, or warm pastel yellow",
        "a fashion item with a mainly yellow tone",
        "a clothing item that is clearly pale yellow or butter yellow rather than beige, cream, tan, or brown",
    ],
    "warm_orange_red": [
        "a clean product photo of a single garment whose dominant clothing color is red, orange red, coral, or warm red",
        "a fashion item with a mainly warm red or orange-red tone",
        "a clothing item that looks clearly warm red, coral, or orange-red rather than brown or pink",
    ],
    "rose_pink": [
        "a clean product photo of a single garment whose dominant clothing color is pink, rose pink, blush pink, or dusty pink",
        "a fashion item with a mainly pink or rose tone",
        "a clothing item that looks pink, blush, or rosy rather than beige or orange red",
    ],
    "natural_green": [
        "a clean product photo of a single garment whose dominant clothing color is green, olive, sage, or natural green",
        "a fashion item with a mainly green tone",
        "a clothing item that reads green, olive, or sage rather than brown or yellow",
    ],
    "fresh_blue": [
        "a clean product photo of a single garment whose dominant clothing color is blue, navy, denim blue, or fresh blue",
        "a fashion item with a mainly blue tone",
        "a clothing item that reads blue or denim rather than gray, black, or purple",
    ],
    "elegant_purple": [
        "a clean product photo of a single garment whose dominant clothing color is purple, lavender, lilac, or plum",
        "a fashion item with a mainly purple tone",
        "a clothing item that appears purple or lavender rather than blue or pink",
    ],
}


OFFICE_OCCASION_CATEGORIES = {
    "shirt",
    "cardigan",
    "blazer",
    "coat",
    "trousers",
    "wide_leg_pants",
    "midi_skirt",
    "midi_dress",
    "heels",
    "flats",
    "boots",
}

SOCIAL_OCCASION_CATEGORIES = {
    "shirt",
    "cardigan",
    "coat",
    "mini_skirt",
    "midi_skirt",
    "mini_dress",
    "midi_dress",
    "boots",
    "sandals",
    "heels",
    "flats",
}

CAMPUS_SOCIAL_BRIDGE_CATEGORIES = {
    "mini_skirt",
    "midi_skirt",
    "mini_dress",
    "midi_dress",
    "boots",
    "sandals",
    "flats",
}

OCCASION_CATEGORY_CAPS: dict[str, int] = {
    "shirt": 2,
    "cardigan": 2,
    "blazer": 2,
    "coat": 2,
    "vest": 2,
    "trousers": 2,
    "wide_leg_pants": 2,
    "mini_skirt": 2,
    "midi_skirt": 2,
    "mini_dress": 2,
    "midi_dress": 2,
    "boots": 2,
    "sandals": 2,
    "heels": 2,
    "flats": 2,
}

SEASON_THREE_LABEL_CATEGORIES = {
    "shirt",
    "cardigan",
    "blazer",
    "coat",
    "vest",
    "windbreaker",
    "jeans",
    "trousers",
    "wide_leg_pants",
    "leggings",
    "denim_jacket",
    "knit_sweater",
    "hoodie",
    "sweatshirt",
}

LIGHT_SEASON_CATEGORIES = {
    "tank_top",
    "shorts",
    "sandals",
    "mini_dress",
    "mini_skirt",
    "heels",
    "flats",
    "t_shirt",
}

HEAVY_SEASON_CATEGORIES = {
    "coat",
    "puffer_jacket",
    "knit_sweater",
    "hoodie",
    "sweatshirt",
    "boots",
    "leggings",
    "beanie",
}

TRANSITIONAL_SEASON_CATEGORIES = {
    "shirt",
    "cardigan",
    "blazer",
    "vest",
    "windbreaker",
    "jeans",
    "trousers",
    "wide_leg_pants",
    "denim_jacket",
    "midi_skirt",
    "midi_dress",
    "sneakers",
}


@dataclass(frozen=True)
class AttributeSelectionConfig:
    min_score: float
    relative_ratio: float
    max_selected: int
    label_min_scores: Mapping[str, float] = field(default_factory=dict)
    second_relative_ratio: float = 0.72
    third_relative_ratio: float = 0.60
    second_max_gap: float = 0.16
    third_max_gap: float = 0.22
    second_strong_score: float = 0.30
    third_strong_score: float = 0.27


OCCASION_SELECTION = AttributeSelectionConfig(
    min_score=0.20,
    relative_ratio=0.60,
    max_selected=2,
    label_min_scores={
        "social": 0.21,
        "campus_casual": 0.21,
        "business_casual": 0.24,
        "professional": 0.27,
    },
    second_relative_ratio=0.72,
    second_max_gap=0.15,
    second_strong_score=0.31,
)

SEASON_SELECTION = AttributeSelectionConfig(
    min_score=0.18,
    relative_ratio=0.58,
    max_selected=3,
    label_min_scores={
        "spring": 0.21,
        "summer": 0.19,
        "autumn": 0.22,
        "winter": 0.20,
    },
    second_relative_ratio=0.69,
    third_relative_ratio=0.56,
    second_max_gap=0.17,
    third_max_gap=0.24,
    second_strong_score=0.30,
    third_strong_score=0.26,
)

OCCASION_FINE_CATEGORY_SECONDARY_PROFILE: dict[tuple[str, str, str], dict[str, float]] = {
    ("professional", "business_casual", "blazer"): {"min_score": 0.44, "min_ratio": 0.93, "max_gap": 0.035},
    ("professional", "business_casual", "trousers"): {"min_score": 0.40, "min_ratio": 0.80, "max_gap": 0.10},
    ("professional", "business_casual", "wide_leg_pants"): {"min_score": 0.35, "min_ratio": 0.80, "max_gap": 0.09},
    ("business_casual", "professional", "shirt"): {"min_score": 0.25, "min_ratio": 0.54, "max_gap": 0.23},
    ("business_casual", "professional", "trousers"): {"min_score": 0.34, "min_ratio": 0.72, "max_gap": 0.13},
    ("business_casual", "professional", "wide_leg_pants"): {"min_score": 0.33, "min_ratio": 0.74, "max_gap": 0.12},
    ("social", "professional", "heels"): {"min_score": 0.10, "min_ratio": 0.16, "max_gap": 0.50},
}

SEASON_FINE_CATEGORY_CAPS: dict[str, int] = {
    "boots": 2,
    "flats": 2,
    "heels": 2,
    "shirt": 3,
    "cardigan": 3,
    "jeans": 3,
    "trousers": 3,
    "midi_skirt": 3,
}

SEASON_FINE_CATEGORY_SECONDARY_PROFILE: dict[tuple[str, str, str], dict[str, float]] = {
    ("winter", "autumn", "boots"): {"min_score": 0.34, "min_ratio": 0.68, "max_gap": 0.17},
    ("winter", "autumn", "knit_sweater"): {"min_score": 0.40, "min_ratio": 0.80, "max_gap": 0.10},
    ("autumn", "spring", "jeans"): {"min_score": 0.26, "min_ratio": 0.62, "max_gap": 0.18, "strong_score": 0.28, "allow_relaxed_thresholds": True},
    ("autumn", "spring", "trousers"): {"min_score": 0.26, "min_ratio": 0.62, "max_gap": 0.18, "strong_score": 0.28, "allow_relaxed_thresholds": True},
    ("spring", "summer", "flats"): {"min_score": 0.30, "min_ratio": 0.74, "max_gap": 0.11},
    ("spring", "autumn", "shirt"): {"min_score": 0.25, "min_ratio": 0.62, "max_gap": 0.18, "strong_score": 0.27, "allow_relaxed_thresholds": True},
    ("spring", "autumn", "cardigan"): {"min_score": 0.26, "min_ratio": 0.62, "max_gap": 0.18, "strong_score": 0.27, "allow_relaxed_thresholds": True},
    ("spring", "autumn", "midi_skirt"): {"min_score": 0.24, "min_ratio": 0.60, "max_gap": 0.18, "strong_score": 0.26, "allow_relaxed_thresholds": True},
}

SEASON_FINE_CATEGORY_THIRD_PROFILE: dict[tuple[str, str, str, str], dict[str, float]] = {
    ("autumn", "spring", "summer", "shirt"): {"min_score": 0.18, "min_ratio": 0.24, "max_gap_from_second": 0.17, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
    ("autumn", "spring", "summer", "midi_skirt"): {"min_score": 0.18, "min_ratio": 0.24, "max_gap_from_second": 0.17, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
    ("spring", "summer", "autumn", "midi_skirt"): {"min_score": 0.30, "min_ratio": 0.62, "max_gap_from_second": 0.16},
    ("spring", "autumn", "winter", "cardigan"): {"min_score": 0.14, "min_ratio": 0.32, "max_gap_from_second": 0.23, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
    ("autumn", "spring", "winter", "blazer"): {"min_score": 0.15, "min_ratio": 0.30, "max_gap_from_second": 0.22, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
    ("autumn", "spring", "winter", "jeans"): {"min_score": 0.12, "min_ratio": 0.24, "max_gap_from_second": 0.24, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
    ("autumn", "spring", "winter", "trousers"): {"min_score": 0.11, "min_ratio": 0.24, "max_gap_from_second": 0.24, "allow_below_label_floor": True, "allow_relaxed_thresholds": True},
}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_prior_map(
    fine_category: str,
    default_prior: Mapping[str, float],
    category_prior_map: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    return dict(category_prior_map.get(fine_category, default_prior))


def _blend_prior_maps(
    primary_map: Mapping[str, float],
    secondary_map: Mapping[str, float] | None = None,
    *,
    primary_weight: float = 0.7,
) -> dict[str, float]:
    if not secondary_map:
        return dict(primary_map)

    secondary_weight = max(0.0, 1.0 - primary_weight)
    merged: dict[str, float] = {}
    for label in set(primary_map.keys()) | set(secondary_map.keys()):
        merged[label] = float(primary_map.get(label, 0.0) * primary_weight + secondary_map.get(label, 0.0) * secondary_weight)
    return merged


def _resolve_occasion_prior_strength(fine_category: str) -> float:
    return float(OCCASION_FINE_CATEGORY_PRIOR_STRENGTH.get(fine_category, 0.18))


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


def _base_selection_debug(config: AttributeSelectionConfig) -> dict[str, Any]:
    return {
        "threshold": config.min_score,
        "relative_ratio": config.relative_ratio,
        "max_selected": config.max_selected,
        "label_min_scores": dict(config.label_min_scores),
        "second_relative_ratio": config.second_relative_ratio,
        "third_relative_ratio": config.third_relative_ratio,
        "second_max_gap": config.second_max_gap,
        "third_max_gap": config.third_max_gap,
        "second_strong_score": config.second_strong_score,
        "third_strong_score": config.third_strong_score,
    }


def _resolve_label_floor(config: AttributeSelectionConfig, label: str) -> float:
    return max(config.min_score, float(config.label_min_scores.get(label, config.min_score)))


def _build_selected_scores(candidates: Sequence[Mapping[str, Any]], selected: Sequence[str]) -> dict[str, float]:
    selected_set = set(selected)
    return {
        str(item["value"]): float(item["score"])
        for item in candidates
        if str(item["value"]) in selected_set
    }


def _is_allowed_occasion_pair(primary: str, secondary: str, fine_category: str, secondary_score: float) -> bool:
    pair = {primary, secondary}

    if pair == {"campus_casual", "professional"}:
        return False

    if pair == {"campus_casual", "business_casual"}:
        return fine_category in {"cardigan", "knit_sweater", "vest", "flats"} and secondary_score >= 0.29

    if pair == {"business_casual", "professional"}:
        return fine_category in OFFICE_OCCASION_CATEGORIES or secondary_score >= 0.33

    if pair == {"social", "professional"}:
        return fine_category in {"blazer", "shirt", "heels", "flats", "coat", "midi_dress", "mini_dress", "boots"} and secondary_score >= 0.30

    if pair == {"business_casual", "social"}:
        return fine_category in SOCIAL_OCCASION_CATEGORIES or secondary_score >= 0.32

    if pair == {"campus_casual", "social"}:
        return fine_category in CAMPUS_SOCIAL_BRIDGE_CATEGORIES or secondary_score >= 0.30

    return True


def _season_category_cap(fine_category: str) -> int:
    if fine_category in SEASON_FINE_CATEGORY_CAPS:
        return int(SEASON_FINE_CATEGORY_CAPS[fine_category])
    return 3 if fine_category in SEASON_THREE_LABEL_CATEGORIES else 2


def _is_allowed_season_pair(primary: str, secondary: str, fine_category: str, secondary_score: float) -> bool:
    pair = {primary, secondary}

    if fine_category in LIGHT_SEASON_CATEGORIES:
        if secondary in {"autumn", "winter"} and primary in {"spring", "summer"}:
            return secondary_score >= 0.33
        if pair == {"spring", "autumn"}:
            return secondary_score >= 0.33

    if fine_category in HEAVY_SEASON_CATEGORIES:
        if secondary == "summer":
            return secondary_score >= 0.34
        if pair == {"spring", "winter"}:
            return False

    if fine_category not in TRANSITIONAL_SEASON_CATEGORIES and pair == {"spring", "autumn"}:
        return secondary_score >= 0.32

    if fine_category not in SEASON_THREE_LABEL_CATEGORIES and secondary == "winter" and primary in {"spring", "autumn"}:
        return secondary_score >= 0.31

    return True




def _resolve_pair_profile(
    *,
    attribute_name: str,
    primary_label: str,
    secondary_label: str,
    fine_category: str,
) -> Mapping[str, float]:
    if attribute_name == "occasion":
        return OCCASION_FINE_CATEGORY_SECONDARY_PROFILE.get((primary_label, secondary_label, fine_category), {})
    if attribute_name == "season":
        return SEASON_FINE_CATEGORY_SECONDARY_PROFILE.get((primary_label, secondary_label, fine_category), {})
    return {}


def _resolve_third_season_profile(
    *,
    primary_label: str,
    secondary_label: str,
    candidate_label: str,
    fine_category: str,
) -> Mapping[str, float]:
    return SEASON_FINE_CATEGORY_THIRD_PROFILE.get((primary_label, secondary_label, candidate_label, fine_category), {})


def _can_add_secondary_label(
    *,
    primary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fine_category: str,
    config: AttributeSelectionConfig,
    attribute_name: str,
) -> tuple[bool, dict[str, Any]]:
    top_score = float(primary["score"])
    candidate_score = float(candidate["score"])
    gap = top_score - candidate_score
    ratio = candidate_score / max(top_score, 1e-6)
    label = str(candidate["value"])

    profile = _resolve_pair_profile(
        attribute_name=attribute_name,
        primary_label=str(primary["value"]),
        secondary_label=label,
        fine_category=fine_category,
    )
    allow_relaxed_thresholds = bool(profile.get("allow_relaxed_thresholds"))
    min_floor = max(_resolve_label_floor(config, label), float(profile.get("min_score", 0.0)))
    resolved_ratio = float(profile.get("min_ratio", config.second_relative_ratio)) if allow_relaxed_thresholds else max(config.second_relative_ratio, float(profile.get("min_ratio", 0.0)))
    resolved_gap = float(profile.get("max_gap", config.second_max_gap)) if allow_relaxed_thresholds else min(config.second_max_gap, float(profile.get("max_gap", config.second_max_gap)))
    resolved_strong_score = float(profile.get("strong_score", config.second_strong_score)) if allow_relaxed_thresholds else max(config.second_strong_score, float(profile.get("strong_score", 0.0)))
    pass_score = candidate_score >= min_floor
    pass_ratio = ratio >= resolved_ratio
    pass_gap = gap <= resolved_gap
    strong_score = candidate_score >= resolved_strong_score

    allowed_pair = True
    if attribute_name == "occasion":
        allowed_pair = _is_allowed_occasion_pair(str(primary["value"]), label, fine_category, candidate_score)
    elif attribute_name == "season":
        allowed_pair = _is_allowed_season_pair(str(primary["value"]), label, fine_category, candidate_score)

    accepted = pass_score and (strong_score or (pass_ratio and pass_gap)) and allowed_pair
    return accepted, {
        "candidate": label,
        "candidate_score": candidate_score,
        "gap_from_top": gap,
        "ratio_to_top": ratio,
        "pass_score": pass_score,
        "pass_ratio": pass_ratio,
        "pass_gap": pass_gap,
        "strong_score": strong_score,
        "allowed_pair": allowed_pair,
        "min_floor": min_floor,
        "resolved_ratio": resolved_ratio,
        "resolved_gap": resolved_gap,
        "resolved_strong_score": resolved_strong_score,
        "profile": dict(profile),
    }


def _can_add_third_season_label(
    *,
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fine_category: str,
    config: AttributeSelectionConfig,
) -> tuple[bool, dict[str, Any]]:
    candidate_score = float(candidate["score"])
    top_score = float(primary["score"])
    second_score = float(secondary["score"])
    ratio_to_top = candidate_score / max(top_score, 1e-6)
    gap_from_second = second_score - candidate_score
    label = str(candidate["value"])

    profile = _resolve_third_season_profile(
        primary_label=str(primary["value"]),
        secondary_label=str(secondary["value"]),
        candidate_label=label,
        fine_category=fine_category,
    )
    base_floor = max(_resolve_label_floor(config, label), config.third_strong_score)
    if profile.get("allow_below_label_floor"):
        min_floor = float(profile.get("min_score", config.third_strong_score))
    else:
        min_floor = max(base_floor, float(profile.get("min_score", config.third_strong_score)))
    allow_relaxed_thresholds = bool(profile.get("allow_relaxed_thresholds"))
    resolved_ratio = float(profile.get("min_ratio", config.third_relative_ratio)) if allow_relaxed_thresholds else max(config.third_relative_ratio, float(profile.get("min_ratio", 0.0)))
    resolved_gap = float(profile.get("max_gap_from_second", config.third_max_gap)) if allow_relaxed_thresholds else min(config.third_max_gap, float(profile.get("max_gap_from_second", config.third_max_gap)))
    pass_score = candidate_score >= min_floor
    pass_ratio = ratio_to_top >= resolved_ratio
    pass_gap = gap_from_second <= resolved_gap
    allowed_pair_primary = _is_allowed_season_pair(str(primary["value"]), label, fine_category, candidate_score)
    allowed_pair_secondary = _is_allowed_season_pair(str(secondary["value"]), label, fine_category, candidate_score)
    accepted = pass_score and pass_ratio and pass_gap and allowed_pair_primary and allowed_pair_secondary
    return accepted, {
        "candidate": label,
        "candidate_score": candidate_score,
        "gap_from_second": gap_from_second,
        "ratio_to_top": ratio_to_top,
        "pass_score": pass_score,
        "pass_ratio": pass_ratio,
        "pass_gap": pass_gap,
        "allowed_pair_primary": allowed_pair_primary,
        "allowed_pair_secondary": allowed_pair_secondary,
        "min_floor": min_floor,
        "resolved_ratio": resolved_ratio,
        "resolved_gap": resolved_gap,
        "profile": dict(profile),
    }


def _select_occasions(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: AttributeSelectionConfig,
    fine_category: str,
) -> dict[str, Any]:
    debug = _base_selection_debug(config)
    if not candidates:
        debug.update({"selected": [], "decision": "empty", "top_score": 0.0, "selected_scores": {}})
        return debug

    primary = candidates[0]
    selected = [str(primary["value"])]
    debug["top_score"] = float(primary["score"])
    debug["category_cap"] = OCCASION_CATEGORY_CAPS.get(fine_category, 1)
    debug["secondary_checks"] = []

    if debug["category_cap"] >= 2 and len(candidates) >= 2:
        accepted, info = _can_add_secondary_label(
            primary=primary,
            candidate=candidates[1],
            fine_category=fine_category,
            config=config,
            attribute_name="occasion",
        )
        debug["secondary_checks"].append(info)
        if accepted:
            selected.append(str(candidates[1]["value"]))

    debug.update(
        {
            "selected": selected,
            "selected_scores": _build_selected_scores(candidates, selected),
            "decision": "threshold_multi" if len(selected) > 1 else "threshold_single",
            "effective_floor": max(config.min_score, debug["top_score"] * config.relative_ratio),
            "max_selected": debug["category_cap"],
        }
    )
    return debug


def _select_seasons(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: AttributeSelectionConfig,
    fine_category: str,
) -> dict[str, Any]:
    debug = _base_selection_debug(config)
    if not candidates:
        debug.update({"selected": [], "decision": "empty", "top_score": 0.0, "selected_scores": {}})
        return debug

    primary = candidates[0]
    selected_items = [primary]
    category_cap = _season_category_cap(fine_category)
    debug["top_score"] = float(primary["score"])
    debug["category_cap"] = category_cap
    debug["secondary_checks"] = []
    debug["third_checks"] = []

    if category_cap >= 2 and len(candidates) >= 2:
        accepted, info = _can_add_secondary_label(
            primary=primary,
            candidate=candidates[1],
            fine_category=fine_category,
            config=config,
            attribute_name="season",
        )
        debug["secondary_checks"].append(info)
        if accepted:
            selected_items.append(candidates[1])

    if category_cap >= 3 and len(selected_items) >= 2 and len(candidates) >= 3:
        accepted, info = _can_add_third_season_label(
            primary=primary,
            secondary=selected_items[1],
            candidate=candidates[2],
            fine_category=fine_category,
            config=config,
        )
        debug["third_checks"].append(info)
        if accepted:
            selected_items.append(candidates[2])

    selected = [str(item["value"]) for item in selected_items]
    debug.update(
        {
            "selected": selected,
            "selected_scores": _build_selected_scores(candidates, selected),
            "decision": "threshold_multi" if len(selected) > 1 else "threshold_single",
            "effective_floor": max(config.min_score, debug["top_score"] * config.relative_ratio),
            "max_selected": category_cap,
        }
    )
    return debug


def infer_occasions(image_features, main_category: str, fine_category: str) -> dict[str, Any]:
    label_map = {value: label for value, label in OCCASION_OPTIONS}
    fine_prior_map = _build_prior_map(fine_category, DEFAULT_OCCASION_PRIOR, OCCASION_PRIORS)
    main_prior_map = MAIN_CATEGORY_OCCASION_PRIORS.get(main_category)
    prior_map = _blend_prior_maps(fine_prior_map, main_prior_map, primary_weight=0.74)

    scored = _score_prompt_ensemble(
        image_features,
        OCCASION_PROMPTS,
        label_map,
        prior_map=prior_map,
        prior_bias_strength=_resolve_occasion_prior_strength(fine_category),
        candidate_temperature=0.06,
    )
    selection = _select_occasions(scored["candidates"], config=OCCASION_SELECTION, fine_category=fine_category)

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
        prior_bias_strength=0.13,
        candidate_temperature=0.055,
    )
    selection = _select_seasons(scored["candidates"], config=SEASON_SELECTION, fine_category=fine_category)

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
        score_map[selected] = 0.82

    if selected == "light_beige":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.32)
        score_map["earth_brown"] = max(score_map["earth_brown"], 0.28)
        score_map["butter_yellow"] = max(score_map["butter_yellow"], 0.24)
    elif selected == "dark_gray_black":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.44)
    elif selected == "neutral_gray":
        score_map["dark_gray_black"] = max(score_map["dark_gray_black"], 0.40)
        score_map["light_beige"] = max(score_map["light_beige"], 0.22)
    elif selected == "earth_brown":
        score_map["light_beige"] = max(score_map["light_beige"], 0.34)
        score_map["butter_yellow"] = max(score_map["butter_yellow"], 0.24)
        score_map["warm_orange_red"] = max(score_map["warm_orange_red"], 0.24)
    elif selected == "butter_yellow":
        score_map["light_beige"] = max(score_map["light_beige"], 0.32)
        score_map["earth_brown"] = max(score_map["earth_brown"], 0.24)
    elif selected == "warm_orange_red":
        score_map["rose_pink"] = max(score_map["rose_pink"], 0.30)
        score_map["earth_brown"] = max(score_map["earth_brown"], 0.24)
    elif selected == "rose_pink":
        score_map["warm_orange_red"] = max(score_map["warm_orange_red"], 0.24)
    elif selected == "fresh_blue":
        score_map["neutral_gray"] = max(score_map["neutral_gray"], 0.22)
    elif selected == "elegant_purple":
        score_map["rose_pink"] = max(score_map["rose_pink"], 0.24)

    return score_map


def _center_crop_image(image: Image.Image, ratio: float = 0.84) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    x1 = max(0, (width - crop_w) // 2)
    y1 = max(0, (height - crop_h) // 2)
    x2 = min(width, x1 + crop_w)
    y2 = min(height, y1 + crop_h)
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def prepare_color_focus_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    detection = detect_main_garment(image)
    if detection.get("detected") and detection.get("bbox"):
        focused = crop_image_by_bbox(image, detection["bbox"], padding_ratio=0.04)
        source = "detected_bbox"
    else:
        focused = _center_crop_image(image, ratio=0.84)
        source = "center_crop_fallback"

    return focused, {
        "source": source,
        "detection": detection,
        "focused_size": list(focused.size),
    }


def _rgb_to_hsv_np(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = arr.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    valid = delta > 1e-6

    r_mask = (cmax == r) & valid
    g_mask = (cmax == g) & valid
    b_mask = (cmax == b) & valid

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    h = h * 60.0

    s = np.zeros_like(cmax)
    nonzero = cmax > 1e-6
    s[nonzero] = delta[nonzero] / cmax[nonzero]
    v = cmax
    return h, s, v


def _estimate_color_stats(image: Image.Image) -> dict[str, float]:
    img = image.convert("RGB")
    width, height = img.size
    scale = min(160 / max(width, height), 1.0)
    resized = img.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    arr = np.asarray(resized)
    if arr.size == 0:
        return {
            "avg_hue": 0.0,
            "avg_sat": 0.0,
            "avg_val": 0.0,
            "light_ratio": 0.0,
            "dark_ratio": 0.0,
            "low_sat_ratio": 0.0,
            "warm_ratio": 0.0,
            "yellow_ratio": 0.0,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
            "blue_ratio": 0.0,
            "cool_gray_ratio": 0.0,
        }

    h, s, v = _rgb_to_hsv_np(arr)
    weights = np.ones_like(v, dtype=np.float32)

    avg_hue = float(np.average(h, weights=weights))
    avg_sat = float(np.average(s, weights=weights))
    avg_val = float(np.average(v, weights=weights))
    total = float(weights.sum())

    def _ratio(mask: np.ndarray) -> float:
        return float(weights[mask].sum() / total) if total > 0 else 0.0

    return {
        "avg_hue": avg_hue,
        "avg_sat": avg_sat,
        "avg_val": avg_val,
        "light_ratio": _ratio(v >= 0.72),
        "dark_ratio": _ratio(v <= 0.25),
        "low_sat_ratio": _ratio(s <= 0.18),
        "warm_ratio": _ratio(((h >= 20) & (h <= 70)) | (h >= 340) | (h < 15)),
        "yellow_ratio": _ratio((h >= 42) & (h <= 72) & (s >= 0.10) & (v >= 0.52)),
        "red_ratio": _ratio((((h >= 350) | (h < 18)) | ((h >= 18) & (h <= 35))) & (s >= 0.18)),
        "green_ratio": _ratio((h >= 70) & (h <= 170) & (s >= 0.12)),
        "blue_ratio": _ratio((h >= 185) & (h <= 270) & (s >= 0.10)),
        "cool_gray_ratio": _ratio((s <= 0.14) & (v >= 0.35) & (v <= 0.75)),
    }


def _merge_color_prior_maps(*maps: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {value: 0.18 for value, _ in COLOR_UI_OPTIONS}
    for score_map in maps:
        for value, score in score_map.items():
            merged[value] = max(float(merged.get(value, 0.18)), float(score))
    return merged


def _build_color_stat_prior_map(stats: Mapping[str, float]) -> dict[str, float]:
    prior = {value: 0.18 for value, _ in COLOR_UI_OPTIONS}

    avg_sat = float(stats.get("avg_sat", 0.0))
    avg_val = float(stats.get("avg_val", 0.0))
    light_ratio = float(stats.get("light_ratio", 0.0))
    dark_ratio = float(stats.get("dark_ratio", 0.0))
    low_sat_ratio = float(stats.get("low_sat_ratio", 0.0))
    warm_ratio = float(stats.get("warm_ratio", 0.0))
    yellow_ratio = float(stats.get("yellow_ratio", 0.0))
    red_ratio = float(stats.get("red_ratio", 0.0))
    green_ratio = float(stats.get("green_ratio", 0.0))
    blue_ratio = float(stats.get("blue_ratio", 0.0))
    cool_gray_ratio = float(stats.get("cool_gray_ratio", 0.0))

    if dark_ratio >= 0.42 and avg_val <= 0.34:
        prior["dark_gray_black"] = 0.92
        prior["neutral_gray"] = max(prior["neutral_gray"], 0.38)

    if low_sat_ratio >= 0.62 and avg_val >= 0.68:
        if warm_ratio >= 0.34:
            prior["light_beige"] = 0.90
            prior["neutral_gray"] = max(prior["neutral_gray"], 0.32)
        else:
            prior["neutral_gray"] = 0.86
            prior["light_beige"] = max(prior["light_beige"], 0.28)

    if cool_gray_ratio >= 0.44 and avg_sat <= 0.18 and 0.36 <= avg_val <= 0.80:
        prior["neutral_gray"] = max(prior["neutral_gray"], 0.88)

    if yellow_ratio >= 0.22 and avg_val >= 0.60:
        prior["butter_yellow"] = 0.94
        prior["light_beige"] = max(prior["light_beige"], 0.42)

    if warm_ratio >= 0.46 and avg_sat <= 0.34 and avg_val <= 0.64:
        prior["earth_brown"] = max(prior["earth_brown"], 0.88)

    if red_ratio >= 0.22 and avg_sat >= 0.18:
        prior["warm_orange_red"] = max(prior["warm_orange_red"], 0.88)

    if green_ratio >= 0.28 and avg_sat >= 0.14 and low_sat_ratio <= 0.72:
        prior["natural_green"] = max(prior["natural_green"], 0.88)

    if blue_ratio >= 0.28 and avg_sat >= 0.14 and low_sat_ratio <= 0.72:
        prior["fresh_blue"] = max(prior["fresh_blue"], 0.88)

    return prior


def _resolve_color_from_stats(stats: Mapping[str, float]) -> str | None:
    avg_sat = float(stats.get("avg_sat", 0.0))
    avg_val = float(stats.get("avg_val", 0.0))
    warm_ratio = float(stats.get("warm_ratio", 0.0))
    yellow_ratio = float(stats.get("yellow_ratio", 0.0))
    red_ratio = float(stats.get("red_ratio", 0.0))
    green_ratio = float(stats.get("green_ratio", 0.0))
    blue_ratio = float(stats.get("blue_ratio", 0.0))
    dark_ratio = float(stats.get("dark_ratio", 0.0))
    low_sat_ratio = float(stats.get("low_sat_ratio", 0.0))
    cool_gray_ratio = float(stats.get("cool_gray_ratio", 0.0))

    if dark_ratio >= 0.45 and avg_val <= 0.34:
        return "dark_gray_black"
    if yellow_ratio >= 0.24 and avg_val >= 0.60:
        return "butter_yellow"
    if green_ratio >= 0.26 and avg_sat >= 0.14 and low_sat_ratio <= 0.72:
        return "natural_green"
    if blue_ratio >= 0.26 and avg_sat >= 0.14 and low_sat_ratio <= 0.72:
        return "fresh_blue"
    if red_ratio >= 0.24 and avg_sat >= 0.18:
        return "warm_orange_red"
    if low_sat_ratio >= 0.62 and avg_val >= 0.70:
        return "light_beige" if warm_ratio >= 0.34 else "neutral_gray"
    if cool_gray_ratio >= 0.44:
        return "neutral_gray"
    return None


def _resolve_ambiguous_color(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stats: Mapping[str, float],
    heuristic_color_value: str | None,
) -> str | None:
    if len(candidates) < 2:
        return None

    top_value = str(candidates[0]["value"])
    second_value = str(candidates[1]["value"])
    gap = float(candidates[0]["score"]) - float(candidates[1]["score"])
    ambiguous_pairs = {
        frozenset({"light_beige", "neutral_gray"}),
        frozenset({"butter_yellow", "light_beige"}),
        frozenset({"butter_yellow", "earth_brown"}),
        frozenset({"earth_brown", "warm_orange_red"}),
        frozenset({"rose_pink", "warm_orange_red"}),
        frozenset({"elegant_purple", "fresh_blue"}),
        frozenset({"fresh_blue", "natural_green"}),
        frozenset({"dark_gray_black", "neutral_gray"}),
    }
    if gap > 0.08 or frozenset({top_value, second_value}) not in ambiguous_pairs:
        return None

    avg_sat = float(stats.get("avg_sat", 0.0))
    avg_val = float(stats.get("avg_val", 0.0))
    low_sat_ratio = float(stats.get("low_sat_ratio", 0.0))
    red_ratio = float(stats.get("red_ratio", 0.0))
    warm_ratio = float(stats.get("warm_ratio", 0.0))

    pair = frozenset({top_value, second_value})
    if pair == frozenset({"earth_brown", "warm_orange_red"}) and low_sat_ratio >= 0.50 and avg_sat <= 0.24 and avg_val <= 0.80:
        return "earth_brown"
    if pair == frozenset({"rose_pink", "warm_orange_red"}) and avg_val >= 0.82 and low_sat_ratio >= 0.52:
        return "rose_pink"
    if pair == frozenset({"butter_yellow", "light_beige"}) and avg_val >= 0.88 and warm_ratio >= 0.90 and red_ratio >= 0.05:
        return "butter_yellow"
    if pair == frozenset({"fresh_blue", "natural_green"}) and avg_sat <= 0.16 and low_sat_ratio >= 0.56:
        return "fresh_blue" if float(stats.get("blue_ratio", 0.0)) >= float(stats.get("green_ratio", 0.0)) else "natural_green"
    if pair == frozenset({"elegant_purple", "fresh_blue"}) and avg_sat <= 0.12 and avg_val >= 0.82:
        return "elegant_purple"
    if pair == frozenset({"light_beige", "neutral_gray"}) and avg_val >= 0.72 and low_sat_ratio >= 0.72 and warm_ratio >= 0.28:
        return "light_beige"

    stat_choice = _resolve_color_from_stats(stats)
    if stat_choice in {top_value, second_value}:
        return stat_choice
    if heuristic_color_value in {top_value, second_value}:
        return heuristic_color_value
    return None


def infer_color(image: Image.Image, *, model_backend: str | None = None) -> dict[str, Any]:
    focused_image, focus_debug = prepare_color_focus_image(image)
    image_features = encode_image_feature(focused_image, model_backend=model_backend)

    heuristic_tone = extract_color(focused_image)
    heuristic_color_value = COLOR_TONE_TO_UI.get(heuristic_tone)
    color_stats = _estimate_color_stats(focused_image)
    heuristic_prior_map = _build_color_prior_map(heuristic_color_value)
    stat_prior_map = _build_color_stat_prior_map(color_stats)
    prior_map = _merge_color_prior_maps(heuristic_prior_map, stat_prior_map)
    label_map = {value: label for value, label in COLOR_UI_OPTIONS}

    prior_bias_strength = 0.12 if focus_debug["source"] == "detected_bbox" else 0.10
    scored = _score_prompt_ensemble(
        image_features,
        COLOR_PROMPTS,
        label_map,
        prior_map=prior_map,
        prior_bias_strength=prior_bias_strength,
        candidate_temperature=0.045,
    )
    candidates = scored["candidates"]

    color_value = str(candidates[0]["value"]) if candidates else (heuristic_color_value or "neutral_gray")
    override_color = _resolve_ambiguous_color(candidates, stats=color_stats, heuristic_color_value=heuristic_color_value)
    if override_color is not None:
        color_value = override_color

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
        "focusDebug": {
            **focus_debug,
            "colorStats": color_stats,
            "statDrivenColor": _resolve_color_from_stats(color_stats),
        },
    }
