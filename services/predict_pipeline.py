from __future__ import annotations

from PIL import Image

from models.clip_model import (
    DEFAULT_MODEL_BACKEND,
    encode_image_feature,
    get_clip_model,
    resolve_backend,
)
from services.classify_category import classify_category
from services.extract_color import extract_color
from services.infer_meta import infer_occasions, infer_seasons
from services.postprocess_category import postprocess_category
from services.validate_input import validate_fashion_input, detect_coarse_fashion_type
from utils.color_tags import build_color_payload
from api_formatters import build_predict_payload


def _resolve_pipeline_backend(model_backend: str | None = None) -> str:
    return resolve_backend(model_backend or DEFAULT_MODEL_BACKEND).key


def run_warmup(model_backend: str | None = None) -> dict:
    backend_key = _resolve_pipeline_backend(model_backend)
    backend_spec = resolve_backend(backend_key)

    try:
        get_clip_model(backend_key)
        dummy = Image.new("RGB", (224, 224), (255, 255, 255))
        image_features = encode_image_feature(dummy, model_backend=backend_key)

        validation = validate_fashion_input(dummy, image_features=image_features, model_backend=backend_key)
        coarse_info = detect_coarse_fashion_type(dummy, image_features=image_features, model_backend=backend_key)
        category_result = classify_category(dummy, image_features=image_features, model_backend=backend_key)
        color_tone = extract_color(dummy)
        color_payload = build_color_payload(color_tone)

        return {
            "ok": True,
            "service": "fashion-attr-service",
            "model": {
                "backend": backend_spec.key,
                "model_name": backend_spec.model_name,
            },
            "warmup": {
                "validation_best_label": validation["best_label"],
                "coarse_type": coarse_info["coarse_type"],
                "category": category_result["categoryKey"],
                "color": color_payload["color"],
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "service": "fashion-attr-service",
            "model": {
                "backend": backend_spec.key,
                "model_name": backend_spec.model_name,
            },
            "error": str(e),
        }


def predict_attributes(original_img, model_backend: str | None = None) -> dict:
    backend_key = _resolve_pipeline_backend(model_backend)

    image_features = encode_image_feature(original_img, model_backend=backend_key)
    validation = validate_fashion_input(original_img, image_features=image_features, model_backend=backend_key)
    if not validation["is_valid"]:
        return {
            "ok": False,
            "reason": "not_fashion_image",
            "validation": {
                "best_label": validation["best_label"],
                "valid_score": validation["valid_score"],
                "invalid_score": validation["invalid_score"],
            },
        }

    route = "product"

    coarse_info = detect_coarse_fashion_type(original_img, image_features=image_features, model_backend=backend_key)

    detection = {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0,
    }

    working_img = original_img

    category_result = classify_category(working_img, image_features=image_features, model_backend=backend_key)
    color_tone = extract_color(working_img)
    color_payload = build_color_payload(color_tone)

    category_result = postprocess_category(
        category_result,
        working_img,
        color_tone=color_tone,
        route=route,
        coarse_info=coarse_info,
        validation=validation,
    )

    occasions = infer_occasions(category_result["mainCategoryKey"], category_result["categoryKey"])
    seasons = infer_seasons(category_result["mainCategoryKey"], category_result["categoryKey"])

    final_score = float(category_result["scores"]["category"])
    if detection["detected"]:
        final_score = float(category_result["scores"]["category"] * 0.75 + detection["score"] * 0.25)

    return build_predict_payload(
        route=route,
        coarse_type=coarse_info["coarse_type"],
        category_result=category_result,
        color_payload=color_payload,
        occasions=occasions,
        seasons=seasons,
        validation=validation,
        detection=detection,
        final_score=final_score,
    )
