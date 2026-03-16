from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

_MODEL = None
_MODEL_LOAD_FAILED = False

PERSON_CLASS_ID = 0
MODEL_NAME = "yolov8n.pt"


def get_yolo_model():
    global _MODEL, _MODEL_LOAD_FAILED

    if _MODEL is not None:
        return _MODEL

    if _MODEL_LOAD_FAILED:
        return None

    try:
        from ultralytics import YOLO

        model_path = Path(MODEL_NAME)

        # 若本地沒有模型檔，ultralytics 通常會自動下載官方權重
        _MODEL = YOLO(str(model_path) if model_path.exists() else MODEL_NAME)
        return _MODEL
    except Exception as e:
        print("get_yolo_model load failed:", e)
        _MODEL_LOAD_FAILED = True
        return None


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def detect_main_subject_bbox(
    image: Image.Image,
    conf: float = 0.25,
) -> dict[str, Any] | None:
    """
    使用 YOLOv8n 偵測 person。
    若模型不可用，直接回傳 None，避免整個 API 啟動失敗。
    """
    model = get_yolo_model()
    if model is None:
        return None

    img = _pil_to_numpy(image)

    try:
        results = model.predict(
            source=img,
            conf=conf,
            verbose=False,
            imgsz=320,
            device="cpu",
        )
    except Exception as e:
        print("detect_main_subject_bbox predict failed:", e)
        return None

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