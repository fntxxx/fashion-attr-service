from __future__ import annotations

from PIL import Image

from fashion_attr_service.api.constants import SERVICE_NAME
from fashion_attr_service.api.exceptions import PredictRejectedError
from fashion_attr_service.api.formatters import build_category_value, build_predict_payload, map_public_value
from fashion_attr_service.api.responses import build_success_response
from fashion_attr_service.api.constants import PUBLIC_COLOR_VALUE_MAP
from fashion_attr_service.models.fashion_siglip_model import (
    BACKEND_SPEC,
    MODEL_BACKEND,
    encode_image_feature,
    get_clip_model,
)
from fashion_attr_service.services.attribute_heads import infer_color, infer_occasions, infer_seasons
from fashion_attr_service.services.classify_category import classify_category
from fashion_attr_service.services.postprocess_category import postprocess_category
from fashion_attr_service.services.validate_input import detect_coarse_fashion_type, validate_fashion_input



def _normalize_pipeline_backend(model_backend: str | None = None) -> str:
    if model_backend is None:
        return MODEL_BACKEND

    backend_key = model_backend.strip().lower()
    if backend_key != MODEL_BACKEND:
        raise ValueError(f"Unsupported model backend: {backend_key}. Supported: {MODEL_BACKEND}")

    return MODEL_BACKEND



def _build_predict_debug_payload(*, pre_category_result: dict, post_category_result: dict, coarse_info: dict, color_payload: dict, occasions: dict, seasons: dict) -> dict:
    return {
        "pre_postprocess_category": {
            "mainCategoryKey": pre_category_result.get("mainCategoryKey"),
            "mainCategory": pre_category_result.get("mainCategory"),
            "categoryKey": pre_category_result.get("categoryKey"),
            "category": pre_category_result.get("category"),
        },
        "postprocess_category": {
            "mainCategoryKey": post_category_result.get("mainCategoryKey"),
            "mainCategory": post_category_result.get("mainCategory"),
            "categoryKey": post_category_result.get("categoryKey"),
            "category": post_category_result.get("category"),
        },
        "coarse_type": coarse_info.get("coarse_type"),
        "coarse_score": float(coarse_info.get("score") or 0.0),
        "coarse_score_map": coarse_info.get("score_map") or {},
        "candidate_score_map": post_category_result.get("candidateScoreMaps") or {},
        "postprocess_debug": post_category_result.get("postprocessDebug") or {},
        "color_focus": color_payload.get("focusDebug") or {},
        "color_heuristic_tone": color_payload.get("heuristicTone"),
        "color_candidate_score_map": color_payload.get("candidateScoreMap") or {},
        "color_raw_prompt_score_map": color_payload.get("rawPromptScoreMap") or {},
        "color_prior_map": color_payload.get("priorMap") or {},
        "occasion_selection": occasions.get("selectionDebug") or {},
        "occasion_candidate_score_map": occasions.get("candidateScoreMap") or {},
        "occasion_raw_prompt_score_map": occasions.get("rawPromptScoreMap") or {},
        "occasion_prior_map": occasions.get("priorMap") or {},
        "occasion_score_bias_map": occasions.get("scoreBiasMap") or {},
        "season_selection": seasons.get("selectionDebug") or {},
        "season_candidate_score_map": seasons.get("candidateScoreMap") or {},
        "season_raw_prompt_score_map": seasons.get("rawPromptScoreMap") or {},
        "season_prior_map": seasons.get("priorMap") or {},
    }




def _build_default_detection() -> dict:
    return {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0,
    }



def _build_detection_from_color_payload(color_payload: dict) -> dict:
    color_focus_detection = (color_payload.get("focusDebug") or {}).get("detection") or {}
    if not color_focus_detection.get("detected"):
        return _build_default_detection()

    return {
        "detected": bool(color_focus_detection.get("detected")),
        "label": color_focus_detection.get("label"),
        "mainCategoryKey": color_focus_detection.get("mainCategoryKey"),
        "bbox": color_focus_detection.get("bbox"),
        "score": float(color_focus_detection.get("score") or 0.0),
    }



def _resolve_final_score(category_score: float, detection: dict) -> float:
    if not detection["detected"]:
        return float(category_score)
    return float(category_score * 0.75 + detection["score"] * 0.25)


def run_warmup(model_backend: str | None = None) -> dict:
    backend_key = _normalize_pipeline_backend(model_backend)

    try:
        get_clip_model(backend_key)
        dummy = Image.new("RGB", (224, 224), (255, 255, 255))
        image_features = encode_image_feature(dummy, model_backend=backend_key)

        validation = validate_fashion_input(dummy, image_features=image_features, model_backend=backend_key)
        coarse_info = detect_coarse_fashion_type(dummy, image_features=image_features, model_backend=backend_key)
        category_result = classify_category(dummy, image_features=image_features, model_backend=backend_key)
        color_payload = infer_color(dummy, model_backend=backend_key)

        return build_success_response(
            {
                "service": SERVICE_NAME,
                "model": {
                    "backend": BACKEND_SPEC.key,
                    "model_name": BACKEND_SPEC.model_name,
                },
                "warmup": {
                    "validation_best_label": validation["best_label"],
                    "coarse_type": coarse_info["coarse_type"],
                    "category": build_category_value(category_result, coarse_type=coarse_info["coarse_type"]),
                    "color": map_public_value(str(color_payload["color"]), PUBLIC_COLOR_VALUE_MAP),
                },
            }
        )
    except Exception as e:
        return {
            "ok": False,
            "service": SERVICE_NAME,
            "model": {
                "backend": BACKEND_SPEC.key,
                "model_name": BACKEND_SPEC.model_name,
            },
            "error": str(e),
        }



def predict_attributes(original_img, model_backend: str | None = None, *, include_debug: bool = False) -> dict:
    backend_key = _normalize_pipeline_backend(model_backend)

    image_features = encode_image_feature(original_img, model_backend=backend_key)
    validation = validate_fashion_input(original_img, image_features=image_features, model_backend=backend_key)
    if not validation["is_valid"]:
        raise PredictRejectedError(
            reason="not_fashion_image",
            validation={
                "best_label": validation["best_label"],
                "valid_score": validation["valid_score"],
                "invalid_score": validation["invalid_score"],
            },
        )

    route = "product"

    coarse_info = detect_coarse_fashion_type(original_img, image_features=image_features, model_backend=backend_key)

    working_img = original_img

    pre_category_result = classify_category(working_img, image_features=image_features, model_backend=backend_key)
    color_payload = infer_color(working_img, model_backend=backend_key)
    detection = _build_detection_from_color_payload(color_payload)

    category_result = postprocess_category(
        dict(pre_category_result),
        working_img,
        color_tone=color_payload["heuristicTone"],
        route=route,
        coarse_info=coarse_info,
        validation=validation,
    )

    occasions = infer_occasions(image_features, category_result["mainCategoryKey"], category_result["categoryKey"])
    seasons = infer_seasons(image_features, category_result["mainCategoryKey"], category_result["categoryKey"])

    final_score = _resolve_final_score(float(category_result["scores"]["category"]), detection)

    payload = build_predict_payload(
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

    if include_debug:
        payload["_debug"] = _build_predict_debug_payload(
            pre_category_result=pre_category_result,
            post_category_result=category_result,
            coarse_info=coarse_info,
            color_payload=color_payload,
            occasions=occasions,
            seasons=seasons,
        )

    return build_success_response(payload)
