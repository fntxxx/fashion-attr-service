from typing import Dict, Any

from models.detector_model import run_detection

# 注意：
# Grounding DINO 的 prompt 需要小寫，且每個查詢詞後面加句點
TEXT_PROMPT = (
    "hat. cap. beanie. "
    "t-shirt. shirt. blouse. polo shirt. hoodie. sweatshirt. sweater. cardigan. "
    "jacket. coat. blazer. vest. tank top. camisole. "
    "pants. jeans. trousers. shorts. skirt. dress. "
    "shoes. sneakers. boots. sandals. heels."
)

LABEL_TO_MAIN_CATEGORY = {
    "hat": "headwear",
    "cap": "headwear",
    "beanie": "headwear",

    "t-shirt": "upper_body",
    "shirt": "upper_body",
    "blouse": "upper_body",
    "polo shirt": "upper_body",
    "hoodie": "upper_body",
    "sweatshirt": "upper_body",
    "sweater": "upper_body",
    "cardigan": "upper_body",
    "jacket": "upper_body",
    "coat": "upper_body",
    "blazer": "upper_body",
    "vest": "upper_body",
    "tank top": "upper_body",
    "camisole": "upper_body",

    "pants": "lower_body",
    "jeans": "lower_body",
    "trousers": "lower_body",
    "shorts": "lower_body",
    "skirt": "lower_body",

    "dress": "dress",

    "shoes": "shoes",
    "sneakers": "shoes",
    "boots": "shoes",
    "sandals": "shoes",
    "heels": "shoes",
}


def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()

    # 基本清理
    label = label.replace(",", " ")
    label = label.replace("/", " ")
    label = label.replace("-", " ")
    label = " ".join(label.split())

    # 常見壞字修正
    replacements = [
        ("tee shirt", "t shirt"),
        ("tee-shirt", "t shirt"),
        ("tshirt", "t shirt"),
        ("pantss", "pants"),
        ("trouserss", "trousers"),
        ("shoees", "shoes"),
        ("dresss", "dress"),
    ]

    for src, dst in replacements:
        label = label.replace(src, dst)

    return label


def _map_to_main_category(label: str):
    label = _normalize_label(label)

    keyword_groups = [
        (["dress"], "dress"),
        (["skirt", "pants", "jeans", "trousers", "shorts"], "lower_body"),
        (
            [
                "top", "upper body clothing", "shirt", "blouse", "polo",
                "hoodie", "sweatshirt", "sweater", "cardigan",
                "jacket", "coat", "blazer", "vest", "tank", "camisole"
            ],
            "upper_body"
        ),
        (["hat", "cap", "beanie"], "headwear"),
        (["shoes", "sneakers", "boots", "sandals", "heels"], "shoes"),
    ]

    for keywords, category in keyword_groups:
        for kw in keywords:
            if kw in label:
                return category

    return None


def _pick_best_detection(image_size, boxes, scores, labels):
    width, height = image_size
    image_area = max(width * height, 1)

    best_item = None
    best_rank = -999.0

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        box_area = box_w * box_h
        area_ratio = box_area / image_area

        center_x = (x1 + x2) / 2.0 / width
        center_y = (y1 + y2) / 2.0 / height
        aspect_ratio = box_h / box_w

        norm_label = _normalize_label(str(label))
        main_category = _map_to_main_category(norm_label)

        if main_category is None:
            continue

        rank = float(score) * 0.60 + float(area_ratio) * 0.25

        # 偏中央加分
        center_bias = 1.0 - abs(center_x - 0.5)
        rank += center_bias * 0.08

        # 太靠下扣分（避免只抓裙擺 / 褲管）
        if center_y > 0.72:
            rank -= 0.12

        # 太矮的框扣分
        if box_h / height < 0.22:
            rank -= 0.10

        # dress 偏好高而直的框
        if main_category == "dress":
            if aspect_ratio > 1.4:
                rank += 0.10
            if box_h / height > 0.45:
                rank += 0.08

        # shoes 偏好靠下
        if main_category == "shoes":
            if center_y > 0.72:
                rank += 0.08

        item = {
            "detected": True,
            "label": norm_label,
            "mainCategoryKey": main_category,
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "score": float(score),
            "rank": float(rank),
            "areaRatio": float(area_ratio)
        }

        if rank > best_rank:
            best_rank = rank
            best_item = item

    return best_item


def detect_main_garment(image) -> Dict[str, Any]:
    try:
        result = run_detection(
            image=image,
            text_prompt=TEXT_PROMPT,
            box_threshold=0.24,
            text_threshold=0.20
        )

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        best = _pick_best_detection(image.size, boxes, scores, labels)

        if best is not None:
            return best

    except Exception as e:
        print("detect_main_garment error:", e)

    return {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0
    }