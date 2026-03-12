from typing import Dict, Any

from models.yolo_detector import detect_main_subject_bbox


def _clip_bbox_to_image(image_size, bbox):
    width, height = image_size
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return [x1, y1, x2, y2]


def _expand_bbox(image_size, bbox, expand_ratio: float = 0.08):
    width, height = image_size
    x1, y1, x2, y2 = bbox

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    pad_x = box_w * expand_ratio
    pad_y = box_h * expand_ratio

    expanded = [
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
    ]
    return _clip_bbox_to_image(image_size, expanded)


def _score_person_bbox(image_size, bbox, score: float) -> float:
    """
    對 YOLO 的 person bbox 做簡單排序分數：
    - 基礎採信 detector score
    - 大框加分
    - 越接近畫面中央加分
    - 太小或太偏下略扣分
    """
    width, height = image_size
    image_area = max(width * height, 1)

    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    box_area = box_w * box_h
    area_ratio = box_area / image_area

    center_x = ((x1 + x2) / 2.0) / max(width, 1)
    center_y = ((y1 + y2) / 2.0) / max(height, 1)

    rank = float(score) * 0.62 + float(area_ratio) * 0.28

    # 越靠近中央越加分
    center_bias_x = 1.0 - abs(center_x - 0.5)
    center_bias_y = 1.0 - abs(center_y - 0.5)
    rank += center_bias_x * 0.06
    rank += center_bias_y * 0.04

    # 太小框扣分
    if box_h / max(height, 1) < 0.25:
        rank -= 0.12

    # 過度貼近底部，通常容易帶太多地板或局部區域
    if center_y > 0.78:
        rank -= 0.08

    return float(rank)


def _pick_best_detection(image_size, detections):
    best_item = None
    best_rank = -999.0

    for det in detections:
        bbox = det.get("bbox")
        score = float(det.get("score", 0.0))
        label = det.get("label")

        if not bbox or len(bbox) != 4:
            continue

        bbox = _clip_bbox_to_image(image_size, bbox)
        rank = _score_person_bbox(image_size, bbox, score)
        bbox = _expand_bbox(image_size, bbox, expand_ratio=0.08)

        width, height = image_size
        x1, y1, x2, y2 = bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(width * height, 1)

        item = {
            "detected": True,
            "label": label,
            "mainCategoryKey": None,
            "bbox": bbox,
            "score": score,
            "rank": rank,
            "areaRatio": float(area_ratio),
        }

        if rank > best_rank:
            best_rank = rank
            best_item = item

    return best_item


def detect_main_garment(image) -> Dict[str, Any]:
    """
    YOLO 版最小穩定整合：

    - 偵測人物主體 bbox
    - 保留 bbox 排序/擴張概念
    - 回傳格式維持與舊版相容

    目前仍不是 clothing-specific detector，
    而是先把人物商品圖聚焦到主體區域，降低背景干擾。
    """

    try:
        bbox = detect_main_subject_bbox(image)

        detections = []
        if bbox is not None:
            detections.append(
                {
                    "label": bbox.get("label", "person"),
                    "bbox": [
                        bbox["x1"],
                        bbox["y1"],
                        bbox["x2"],
                        bbox["y2"],
                    ],
                    "score": float(bbox.get("score", 0.0)),
                }
            )

        best = _pick_best_detection(image.size, detections)

        if best is not None:
            return {
                "detected": True,
                "label": best.get("label"),
                "mainCategoryKey": best.get("mainCategoryKey"),
                "bbox": best.get("bbox"),
                "score": float(best.get("score", 0.0)),
            }

    except Exception as e:
        print("detect_main_garment error:", e)

    return {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0,
    }