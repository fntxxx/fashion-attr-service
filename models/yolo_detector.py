from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

_MODEL: Optional[YOLO] = None

PERSON_CLASS_ID = 0


def get_yolo_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def detect_main_subject_bbox(
    image: Image.Image,
    conf: float = 0.25,
) -> dict[str, Any] | None:
    """
    第一版最小可行 detector：
    - 使用 YOLOv8n 偵測 person
    - 若有多個 person，取面積最大的
    - 若沒偵測到 person，回傳 None

    這不是最終 clothing-specific detector，
    而是先讓人物商品圖能先聚焦到主體區域，
    降低背景與非主衣物干擾。
    """
    model = get_yolo_model()
    img = _pil_to_numpy(image)

    results = model.predict(
        source=img,
        conf=conf,
        verbose=False,
        imgsz=640,
        device="cpu",
    )

    if not results:
        return None

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_bbox: dict[str, Any] | None = None
    best_area = -1.0

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        if cls_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h

        if area > best_area:
            best_area = area
            best_bbox = {
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "score": score,
                "label": "person",
            }

    return best_bbox