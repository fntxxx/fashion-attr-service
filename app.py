from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from services.detect_garment import detect_main_garment
from services.crop_garment import crop_image_by_bbox
from services.classify_category import classify_category
from services.extract_color import extract_color
from services.infer_meta import infer_style, infer_season
from services.postprocess_category import postprocess_category
from services.validate_input import (
    validate_fashion_input,
    detect_image_route,
    detect_coarse_fashion_type,
)
from utils.color_tags import build_color_payload

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    original_img = Image.open(io.BytesIO(contents)).convert("RGB")

    validation = validate_fashion_input(original_img)

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
    coarse_info = detect_coarse_fashion_type(original_img)

    detection = {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0,
    }

    working_img = original_img

    category_result = classify_category(working_img)
    color_tone = extract_color(working_img)
    color_payload = build_color_payload(color_tone)

    category_result = postprocess_category(
        category_result,
        working_img,
        color_tone=color_tone,
        route=route,
        coarse_info=coarse_info,
        validation=validation
    )

    style = infer_style(
        category_result["mainCategoryKey"],
        category_result["categoryKey"]
    )
    season = infer_season(
        category_result["mainCategoryKey"],
        category_result["categoryKey"]
    )

    final_score = category_result["scores"]["category"]

    if detection["detected"]:
        final_score = (
            category_result["scores"]["category"] * 0.75
            + detection["score"] * 0.25
        )

    return {
        "ok": True,
        "route": route,
        "coarseType": coarse_info["coarse_type"],

        "mainCategory": category_result["mainCategory"],
        "mainCategoryKey": category_result["mainCategoryKey"],
        "category": category_result["category"],
        "categoryKey": category_result["categoryKey"],
        "colorTone": color_payload["colorTone"],
        "colorTags": color_payload["colorTags"],
        "style": style,
        "season": season,

        "scores": {
            "mainCategory": category_result["scores"]["mainCategory"],
            "category": category_result["scores"]["category"],
            "occasion": float(final_score),
            "colorTone": float(final_score),
            "season": float(final_score)
        },
        "score": float(final_score),

        "detected": detection["detected"],
        "detectedLabel": detection["label"],
        "bbox": detection["bbox"],

        "validation": {
            "best_label": validation["best_label"],
            "valid_score": validation["valid_score"],
            "invalid_score": validation["invalid_score"],
        }
    }