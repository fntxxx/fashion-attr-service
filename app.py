from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from services.classify_category import classify_category
from services.extract_color import extract_color
from services.infer_meta import (
    infer_style,
    infer_season,
    infer_occasions,
    infer_seasons,
)
from services.postprocess_category import postprocess_category
from services.validate_input import (
    validate_fashion_input,
    detect_coarse_fashion_type,
)
from utils.color_tags import build_color_payload

app = FastAPI()


CATEGORY_UI_OPTIONS = [
    ("top", "上衣"),
    ("pants", "褲子"),
    ("skirt", "裙子"),
    ("dress", "連身裙"),
    ("outer", "外套"),
    ("shoes", "鞋子"),
]

MAIN_CATEGORY_TO_UI = {
    "upper_body": "top",
    "pants": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "shoes": "shoes",
    # 目前 UI 沒有帽子，先保守 fallback 到 top
    "headwear": "top",
}

OUTER_FINE_KEYS = {
    "denim_jacket",
    "blazer",
    "coat",
    "puffer_jacket",
    "vest",
    "windbreaker",
}


def build_category_selection(category_result: dict) -> dict:
    fine_key = category_result["categoryKey"]
    main_key = category_result["mainCategoryKey"]

    if fine_key in OUTER_FINE_KEYS:
        selected = "outer"
    else:
        selected = MAIN_CATEGORY_TO_UI.get(main_key, "top")

    label_map = {value: label for value, label in CATEGORY_UI_OPTIONS}
    main_score = float(category_result["scores"]["mainCategory"])
    fine_score = float(category_result["scores"]["category"])
    selected_score = max(main_score, fine_score)

    candidates = []
    for value, label in CATEGORY_UI_OPTIONS:
        score = 0.03
        if value == selected:
            score = selected_score
        elif selected == "top" and value == "outer":
            score = 0.18 if fine_key not in OUTER_FINE_KEYS else 0.32
        elif selected == "outer" and value == "top":
            score = 0.20
        elif selected in {"skirt", "dress"} and value in {"skirt", "dress"} and value != selected:
            score = 0.14
        elif selected == "pants" and fine_key in {"wide_leg_pants", "shorts"} and value == "skirt":
            score = 0.08
        candidates.append({
            "value": value,
            "label": label,
            "score": float(score),
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return {
        "selected": selected,
        "label": label_map[selected],
        "score": float(selected_score),
        "candidates": candidates,
    }


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
        validation=validation,
    )

    legacy_style = infer_style(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    legacy_season = infer_season(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )

    occasions = infer_occasions(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    seasons = infer_seasons(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    category_selection = build_category_selection(category_result)

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

        # 舊欄位保留
        "mainCategory": category_result["mainCategory"],
        "mainCategoryKey": category_result["mainCategoryKey"],
        "category": category_result["category"],
        "categoryKey": category_result["categoryKey"],
        "colorTone": color_payload["colorTone"],
        "colorTags": color_payload["colorTags"],
        "style": legacy_style,
        "season": legacy_season,

        # 新欄位
        "categorySelection": category_selection,
        "occasions": {
            "selected": occasions["selected"],
            "candidates": occasions["candidates"],
            "threshold": 0.62,
            "maxSelected": 2,
        },
        "seasons": {
            "selected": seasons["selected"],
            "candidates": seasons["candidates"],
            "threshold": 0.58,
            "maxSelected": 2,
        },
        "colors": color_payload["colors"],

        "scores": {
            "mainCategory": category_result["scores"]["mainCategory"],
            "category": category_result["scores"]["category"],
            "occasion": float(max([x["score"] for x in occasions["candidates"]] or [0.0])),
            "colorTone": float(max([x["score"] for x in color_payload["colors"]["candidates"]] or [0.0])),
            "season": float(max([x["score"] for x in seasons["candidates"]] or [0.0])),
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