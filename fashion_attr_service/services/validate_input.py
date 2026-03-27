from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from fashion_attr_service.models.fashion_siglip_model import (
    predict_topk,
    score_texts,
    encode_image_feature,
    predict_topk_with_image_feature,
    score_texts_with_image_feature,
)
from fashion_attr_service.models.yolo_detector import detect_main_subject_bbox


VALID_LABELS = [
    "a clean product photo of a single clothing item on a plain background",
    "a catalog photo of a single garment",
    "a studio product photo of one clothing item",
    "a front-view product image of a single garment",
    "a single clothing item centered on a clean background",
    "a single fashion item laid flat on a plain background",
    "a flat lay photo of one garment only",
    "a single isolated apparel item",
    "a product image of one shirt",
    "a product image of one jacket",
    "a product image of one pair of pants",
    "a product image of one skirt or one dress",
    "a product image of one pair of shoes",
    "a product image of one hat",
]

INVALID_LABEL_GROUPS = {
    "person": [
        "a photo of a person wearing clothes",
        "a full-body outfit photo with a person",
        "a street style outfit photo",
        "a lifestyle fashion photo with a model",
        "a fashion lookbook image with a person",
        "a mirror selfie outfit photo",
        "a person standing and showing an outfit",
        "a person wearing a dress",
        "a person wearing a shirt and pants",
        "a fashion photo with visible face and body",
    ],
    "multi_item": [
        "a photo with multiple clothing items",
        "multiple garments in one image",
        "several clothing items laid together",
        "an outfit set with top and bottom together",
        "a clothing stack",
        "a pile of clothes",
        "a wardrobe or closet scene with many clothes",
        "a clothing rack with multiple garments",
        "a photo showing two or more garments",
        "a flat lay with multiple fashion items",
    ],
    "non_fashion": [
        "a landscape photo",
        "a food photo",
        "a pet photo",
        "a photo of furniture",
        "a photo of electronics",
        "a photo of a car",
        "a screenshot of a website",
        "a poster or graphic design image",
        "a document or paper",
        "a close-up face photo",
        "an animal photo",
        "a building or room photo",
    ],
    "messy": [
        "a messy background product photo",
        "a cluttered room photo",
        "a busy background photo",
        "a lifestyle indoor snapshot",
    ],
}

INVALID_LABELS = [
    label
    for labels in INVALID_LABEL_GROUPS.values()
    for label in labels
]


@dataclass
class PromptScore:
    best_label: str
    best_score: float
    results: list[dict[str, Any]]


def _score_label_group_softmax(image_features, labels, topk=5, model_backend: str | None = None) -> PromptScore:
    results = predict_topk_with_image_feature(
        image_features,
        labels,
        topk=min(topk, len(labels)),
        model_backend=model_backend,
    )

    best = results[0]

    return PromptScore(
        best_label=best["label"],
        best_score=float(best["score"]),
        results=results,
    )


def _score_label_group_raw(image_features, labels, topk=5, model_backend: str | None = None) -> PromptScore:
    results = sorted(
        score_texts_with_image_feature(image_features, labels, model_backend=model_backend),
        key=lambda x: x["score"],
        reverse=True,
    )
    best = results[0]
    return PromptScore(
        best_label=best["label"],
        best_score=float(best["score"]),
        results=results[: min(topk, len(results))],
    )


def _estimate_foreground_components(image) -> dict[str, float]:
    """
    用便宜的影像啟發式估前景塊數。
    用途：
    - 輔助 multi-item 判斷
    - 不追求精準分割，只提供結構訊號
    """
    img = image.convert("RGB").resize((256, 256))
    arr = np.array(img).astype(np.int16)

    border_pixels = np.concatenate(
        [
            arr[:16, :, :].reshape(-1, 3),
            arr[-16:, :, :].reshape(-1, 3),
            arr[:, :16, :].reshape(-1, 3),
            arr[:, -16:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = border_pixels.mean(axis=0)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2)).astype(np.uint8)

    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = mask.shape[0] * mask.shape[1]

    large_components = 0
    component_area_ratios: list[float] = []
    boxes: list[dict[str, int]] = []

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        area_ratio = area / image_area
        if area_ratio < 0.012:
            continue

        large_components += 1
        component_area_ratios.append(float(area_ratio))
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        boxes.append({"x": x, "y": y, "w": w, "h": h})

    boxes = sorted(boxes, key=lambda b: b["w"] * b["h"], reverse=True)

    separation_ratio = 0.0
    if len(boxes) >= 2:
        b1, b2 = boxes[0], boxes[1]
        c1x = b1["x"] + b1["w"] / 2
        c2x = b2["x"] + b2["w"] / 2
        separation_ratio = abs(c1x - c2x) / 256.0

    return {
        "large_components": int(large_components),
        "largest_component_ratio": float(max(component_area_ratios) if component_area_ratios else 0.0),
        "separation_ratio": float(separation_ratio),
    }


def validate_fashion_input(image, image_features=None, model_backend: str | None = None):
    if image_features is None:
        image_features = encode_image_feature(image, model_backend=model_backend)
    """
    嚴格模式：
    只接受「單件衣物 + 乾淨背景 + 沒有人 + 非多件」圖片。

    這版修正重點：
    1. 保留 raw score 做主要判斷。
    2. person 判斷改成「YOLO + CLIP corroboration」，
       避免商品圖被 YOLO 誤判成 person 後直接擋掉。
    3. 加入 valid rescue rule：
       若 valid 明顯強於 invalid，優先保留正常商品圖/平拍圖。
    """
    valid_softmax = _score_label_group_softmax(image_features, VALID_LABELS, topk=5, model_backend=model_backend)
    invalid_softmax = _score_label_group_softmax(image_features, INVALID_LABELS, topk=5, model_backend=model_backend)

    valid_raw = _score_label_group_raw(image_features, VALID_LABELS, topk=5, model_backend=model_backend)

    invalid_raw_groups = {
        group_name: _score_label_group_raw(image_features, labels, topk=3, model_backend=model_backend)
        for group_name, labels in INVALID_LABEL_GROUPS.items()
    }

    valid_max = float(valid_raw.best_score)
    invalid_max = float(max(group.best_score for group in invalid_raw_groups.values()))
    margin = valid_max - invalid_max

    person_clip_score = float(invalid_raw_groups["person"].best_score)
    multi_item_clip_score = float(invalid_raw_groups["multi_item"].best_score)
    non_fashion_clip_score = float(invalid_raw_groups["non_fashion"].best_score)
    messy_clip_score = float(invalid_raw_groups["messy"].best_score)

    component_info = _estimate_foreground_components(image)
    large_components = int(component_info["large_components"])
    separation_ratio = float(component_info["separation_ratio"])

    person_bbox = None
    person_detected = False
    person_score = 0.0
    person_area_ratio = 0.0

    try:
        person_bbox = detect_main_subject_bbox(image, conf=0.25)
        if person_bbox is not None:
            person_score = float(person_bbox.get("score", 0.0))

            x1 = int(person_bbox["x1"])
            y1 = int(person_bbox["y1"])
            x2 = int(person_bbox["x2"])
            y2 = int(person_bbox["y2"])

            width, height = image.size
            box_area = max(1, (x2 - x1) * (y2 - y1))
            image_area = max(1, width * height)
            person_area_ratio = box_area / image_area

            # 先只記錄「有一定人形框跡象」
            # 不直接當成最終 strong person signal
            person_detected = bool(
                (person_score >= 0.30 and person_area_ratio >= 0.10) or
                (person_score >= 0.40 and person_area_ratio >= 0.06) or
                (person_area_ratio >= 0.22 and person_score >= 0.22)
            )
    except Exception as e:
        print("validate_fashion_input person detection error:", e)

    # --------
    # Rescue rule
    # --------
    # 這是針對目前 5 張誤判案例的核心修正：
    # 若 valid 很明顯比 invalid 強，而且 person CLIP 並不接近 valid，
    # 就不該因為 YOLO 的弱誤判直接擋掉。
    valid_rescue = bool(
        margin >= 0.01 and
        person_clip_score < valid_max and
        non_fashion_clip_score < valid_max and
        multi_item_clip_score < valid_max
    )

    # --------
    # Person signal
    # --------
    # 改成較保守：
    # 1. 很大的 person 框 + 不低的 score
    # 2. YOLO 與 person CLIP 同時支持
    strong_person_signal = bool(
        (
            person_detected and
            person_score >= 0.42 and
            person_area_ratio >= 0.18
        ) or
        (
            person_detected and
            person_score >= 0.30 and
            person_area_ratio >= 0.10 and
            person_clip_score >= valid_max - 0.001
        )
    )

    # 單件商品圖保護：
    # 若畫面只有單一明顯主體、valid 分數夠高，且 person bbox 沒大到像完整人像，
    # 就不要因為弱 person 訊號直接擋掉
    single_item_rescue = bool(
        margin >= 0.015 and
        large_components == 1 and
        person_area_ratio <= 0.35
    )

    # rescue 成立時，覆蓋掉弱人物誤判
    if (valid_rescue or single_item_rescue) and person_clip_score < valid_max:
        strong_person_signal = False

    strong_multi_item_signal = bool(
        (multi_item_clip_score > valid_max + 0.0020) or
        (
            multi_item_clip_score >= valid_max - 0.0015 and
            large_components >= 3
        ) or
        (
            multi_item_clip_score >= valid_max - 0.0008 and
            large_components >= 2 and
            separation_ratio >= 0.20
        )
    )

    strong_non_fashion_signal = bool(
        non_fashion_clip_score > valid_max + 0.0020
    )

    strong_messy_signal = bool(
        messy_clip_score > valid_max + 0.0035
    )

    is_valid_by_clip = bool(
        valid_max >= invalid_max - 0.005
    )

    is_valid = bool(
        (is_valid_by_clip or valid_rescue) and
        not strong_person_signal and
        not strong_multi_item_signal and
        not strong_non_fashion_signal and
        not strong_messy_signal
    )

    top_matches = sorted(
        valid_softmax.results + invalid_softmax.results,
        key=lambda x: x["score"],
        reverse=True,
    )[:6]

    if strong_person_signal:
        best_label = "person_detected"
        best_score = max(person_score, person_clip_score)
    elif strong_multi_item_signal:
        best_label = invalid_raw_groups["multi_item"].best_label
        best_score = multi_item_clip_score
    elif strong_non_fashion_signal:
        best_label = invalid_raw_groups["non_fashion"].best_label
        best_score = non_fashion_clip_score
    elif strong_messy_signal:
        best_label = invalid_raw_groups["messy"].best_label
        best_score = messy_clip_score
    else:
        best_label = (
            valid_raw.best_label
            if valid_max >= invalid_max
            else max(
                invalid_raw_groups.items(),
                key=lambda item: item[1].best_score,
            )[1].best_label
        )
        best_score = max(valid_max, invalid_max)

    return {
        "is_valid": bool(is_valid),
        "best_label": best_label,
        "best_score": float(best_score),
        "valid_score": float(valid_max),
        "invalid_score": float(invalid_max),
        "margin": float(margin),
        "top_matches": top_matches,
        "input_policy": "single_clean_garment_only",
        "person_detected": bool(person_detected),
        "person_score": float(person_score),
        "person_area_ratio": float(person_area_ratio),
        "person_bbox": (
            [
                int(person_bbox["x1"]),
                int(person_bbox["y1"]),
                int(person_bbox["x2"]),
                int(person_bbox["y2"]),
            ]
            if person_bbox is not None
            else None
        ),
        "multi_item_detected": bool(strong_multi_item_signal),
        "large_components": int(large_components),
        "component_separation_ratio": float(separation_ratio),
        "raw_scores": {
            "valid": float(valid_max),
            "person": float(person_clip_score),
            "multi_item": float(multi_item_clip_score),
            "non_fashion": float(non_fashion_clip_score),
            "messy": float(messy_clip_score),
        },
    }


def detect_image_route(image, model_backend: str | None = None):
    """
    在嚴格單件衣物模式下，通過驗證的圖片一律當作 product。
    這個函式先保留，避免 app.py 其他地方暫時壞掉。
    """

    image_features = encode_image_feature(image, model_backend=model_backend)

    result = _score_label_group_softmax(image_features, VALID_LABELS, topk=5, model_backend=model_backend)

    return {
        "route": "product",
        "score": float(result.best_score),
        "product_score": float(result.best_score),
        "outfit_score": 0.0,
        "best_label": result.best_label,
        "top_matches": result.results[:5],
    }


def detect_coarse_fashion_type(image, image_features=None, model_backend: str | None = None):
    prompt_groups = {
        "pants": [
            "a product photo of pants",
            "a product photo of trousers",
            "a lower-body garment with two separate pant legs",
            "wide-leg pants laid flat",
            "jeans or trousers product photo",
        ],
        "skirt": [
            "a product photo of a skirt",
            "a lower-body garment without separated legs",
            "a straight skirt product photo",
            "a midi skirt laid flat",
            "a long skirt product photo",
            "a denim skirt product photo",
            "a skirt with a single continuous lower panel and no separated legs",
        ],
        "dress": [
            "a product photo of a dress",
            "a one-piece dress garment",
            "a dress with a connected top and skirt",
            "a midi dress product photo",
            "a long dress laid flat",
        ],
        "upper_body": [
            "a product photo of a top",
            "an upper-body garment product photo",
            "a shirt, blouse, t-shirt, hoodie, sweater, or jacket",
            "a top clothing item laid flat",
            "a product image of upper-body clothing",
        ],
        "headwear": [
            "a product photo of a hat",
            "a cap or beanie product image",
            "a headwear item laid flat",
            "a fashion hat product photo",
            "a product image of headwear",
        ],
        "shoes": [
            "a product photo of shoes",
            "a pair of sneakers, boots, sandals, or heels",
            "footwear product image",
            "a pair of shoes laid flat",
            "a fashion footwear product photo",
        ],
    }

    flat_prompts = []
    prompt_to_group = {}

    for group_key, prompts in prompt_groups.items():
        for prompt in prompts:
            flat_prompts.append(prompt)
            prompt_to_group[prompt] = group_key

    if image_features is None:
        image_features = encode_image_feature(image, model_backend=model_backend)

    results = score_texts_with_image_feature(image_features, flat_prompts, model_backend=model_backend)

    grouped_scores = {key: [] for key in prompt_groups.keys()}
    for item in results:
        prompt = item["label"]
        score = float(item["score"])
        group_key = prompt_to_group[prompt]
        grouped_scores[group_key].append(score)

    score_map = {}
    for group_key, scores in grouped_scores.items():
        scores = sorted(scores, reverse=True)
        top_scores = scores[:2] if len(scores) >= 2 else scores
        score_map[group_key] = float(sum(top_scores) / len(top_scores))

    best_key = max(score_map, key=score_map.get)

    top_matches = sorted(
        [
            {
                "coarse_type": prompt_to_group[item["label"]],
                "prompt": item["label"],
                "score": float(item["score"]),
            }
            for item in results
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:6]

    return {
        "coarse_type": best_key,
        "score": float(score_map[best_key]),
        "score_map": score_map,
        "top_matches": top_matches,
    }