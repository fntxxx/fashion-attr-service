from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io
import colorsys
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.model = None
app.state.processor = None


# =========================
# CLIP texts
# =========================
# 每個 tuple:
# (key, zh, prompt_text)

CATEGORY_TEXTS: List[Tuple[str, str, str]] = [
    ("t-shirt", "T 恤", "a product photo of a short-sleeve t-shirt"),
    ("shirt", "襯衫", "a product photo of a button-up shirt"),
    ("hoodie", "帽T", "a product photo of a hooded sweatshirt"),
    ("sweatshirt", "大學T", "a product photo of a crewneck sweatshirt"),
    ("sweater", "毛衣", "a product photo of a knitted sweater"),
    ("jacket", "外套", "a product photo of a casual jacket"),
    ("coat", "大衣", "a product photo of a long coat"),
    ("dress", "洋裝", "a product photo of a dress"),
    ("skirt", "裙子", "a product photo of a skirt"),
    ("jeans", "牛仔褲", "a product photo of denim jeans"),
    ("pants", "長褲", "a product photo of casual pants"),
    ("shorts", "短褲", "a product photo of shorts"),
]

OCCASION_TEXTS: List[Tuple[str, str, str]] = [
    ("casual", "日常休閒", "this clothing item is suitable for casual daily wear"),
    ("office", "上班通勤", "this clothing item is suitable for office wear"),
    ("formal", "正式場合", "this clothing item is suitable for formal occasions"),
    ("outdoor", "戶外活動", "this clothing item is suitable for outdoor activities"),
]

SEASON_TEXTS: List[Tuple[str, str, str]] = [
    ("summer", "夏季", "this clothing item is best for hot weather"),
    ("mild", "春秋", "this clothing item is best for mild weather"),
    ("winter", "冬季", "this clothing item is best for cold weather"),
    ("all-season", "四季皆可", "this clothing item is suitable for all-season wear"),
]


# =========================
# Model lazy load
# =========================
def get_clip():
    if app.state.model is None or app.state.processor is None:
        app.state.model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        app.state.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        app.state.model.eval()
    return app.state.model, app.state.processor


# =========================
# Image helpers
# =========================
def _open_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _resize_for_color(image: Image.Image, max_side: int = 256) -> Image.Image:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size)


# =========================
# CLIP ranking
# =========================
def _rank_top1(
    image: Image.Image,
    texts: List[Tuple[str, str, str]],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> Dict[str, Any]:
    keys = [item[0] for item in texts]
    zh_map = {item[0]: item[1] for item in texts}
    prompt_texts = [item[2] for item in texts]

    inputs = processor(
        text=prompt_texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)

    idx = int(torch.argmax(probs).item())
    top_score = float(probs[idx].item())
    top_key = keys[idx]
    top_zh = zh_map[top_key]

    score_map = {
        keys[i]: float(probs[i].item())
        for i in range(len(keys))
    }

    return {
        "key": top_key,
        "zh": top_zh,
        "score": top_score,
        "score_map": score_map,
    }


# =========================
# Color detection by pixels
# =========================
def _extract_valid_pixels(image: Image.Image) -> np.ndarray:
    img = _resize_for_color(image)
    rgba = img.convert("RGBA")
    arr = np.array(rgba)

    rgb = arr[:, :, :3].reshape(-1, 3)
    alpha = arr[:, :, 3].reshape(-1)

    # 若有透明背景，優先用 alpha 過濾
    if np.any(alpha < 250):
        mask = alpha > 20
        pixels = rgb[mask]
        if len(pixels) > 0:
            return pixels

    # 若沒有透明資訊，退回全部像素
    return rgb


def _rgb_to_color_family(r: int, g: int, b: int) -> str:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)

    # 白 / 黑 / 灰 / 米色優先判
    if v >= 0.88 and s <= 0.10:
        return "白色系"

    if v <= 0.18:
        return "黑色系"

    if s <= 0.12 and 0.18 < v < 0.88:
        return "灰色系"

    # 米色：亮度偏高、飽和不高、偏黃橘
    if v >= 0.65 and s <= 0.28 and (0.08 <= h <= 0.16):
        return "米色系"

    # 彩色區間
    if (h >= 0.95 or h < 0.04):
        return "紅色系"

    if 0.04 <= h < 0.12:
        return "棕色系"

    if 0.12 <= h < 0.20:
        return "米色系"

    if 0.20 <= h < 0.45:
        return "綠色系"

    if 0.45 <= h < 0.75:
        return "藍色系"

    return "灰色系"


def _detect_color_tone(image: Image.Image) -> Dict[str, Any]:
    pixels = _extract_valid_pixels(image)
    if len(pixels) == 0:
        return {
            "zh": "灰色系",
            "score": 0.0,
        }

    # 避免極端雜訊，使用中位數顏色
    median_rgb = np.median(pixels, axis=0).astype(np.uint8)
    r, g, b = int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2])

    family = _rgb_to_color_family(r, g, b)

    # 用與中位數相近像素比例，當成簡單信心值
    dist = np.linalg.norm(pixels.astype(np.int16) - median_rgb.astype(np.int16), axis=1)
    score = float(np.mean(dist < 40))
    score = max(0.0, min(1.0, score))

    return {
        "zh": family,
        "score": score,
    }


# =========================
# Rule-based corrections
# =========================
def _normalize_category(category: Dict[str, Any]) -> Dict[str, Any]:
    # 目前先保留 category 原始 top1
    return category


def _normalize_occasion(category_key: str, occasion: Dict[str, Any]) -> Dict[str, Any]:
    casual_keys = {"t-shirt", "hoodie", "sweatshirt", "jeans", "pants", "shorts", "skirt"}
    formal_keys = {"shirt", "dress", "coat"}

    # 分數不高時做收斂，降低亂跳
    if occasion["score"] < 0.60:
        if category_key in casual_keys:
            occasion["key"] = "casual"
            occasion["zh"] = "日常休閒"
        elif category_key in formal_keys:
            if occasion["key"] != "outdoor":
                occasion["key"] = "office"
                occasion["zh"] = "上班通勤"

    return occasion


def _normalize_season(category_key: str, season: Dict[str, Any]) -> Dict[str, Any]:
    # 分數不高時，用類別做保守修正
    if season["score"] < 0.60:
        if category_key in {"coat"}:
            season["key"] = "winter"
            season["zh"] = "冬季"
        elif category_key in {"shorts", "t-shirt"}:
            season["key"] = "summer"
            season["zh"] = "夏季"
        elif category_key in {"hoodie", "sweatshirt", "sweater", "jacket"}:
            season["key"] = "mild"
            season["zh"] = "春秋"

    return season


# =========================
# API
# =========================
@app.get("/")
def root():
    return {
        "service": "fashion-attr-service",
        "ok": True,
        "mode": "lazy-load-cpu",
        "model": MODEL_ID,
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    original_image = _open_image(raw)
    rgb_image = _to_rgb(original_image)

    model, processor = get_clip()

    category = _rank_top1(
        image=rgb_image,
        texts=CATEGORY_TEXTS,
        model=model,
        processor=processor,
    )
    category = _normalize_category(category)

    occasion = _rank_top1(
        image=rgb_image,
        texts=OCCASION_TEXTS,
        model=model,
        processor=processor,
    )
    occasion = _normalize_occasion(category["key"], occasion)

    season = _rank_top1(
        image=rgb_image,
        texts=SEASON_TEXTS,
        model=model,
        processor=processor,
    )
    season = _normalize_season(category["key"], season)

    # colorTone 改走像素法
    color_tone = _detect_color_tone(original_image)

    return {
        "category": category["zh"],
        "occasion": occasion["zh"],
        "colorTone": color_tone["zh"],
        "season": season["zh"],
        "scores": {
            "category": category["score"],
            "occasion": occasion["score"],
            "colorTone": color_tone["score"],
            "season": season["score"],
        },
        "received": {
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(raw),
        },
    }