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

    ("sneakers", "運動鞋", "a product photo of sneakers"),
    ("boots", "靴子", "a product photo of boots"),
    ("sandals", "涼鞋", "a product photo of sandals"),
    ("heels", "高跟鞋", "a product photo of high heels"),
    ("loafers", "樂福鞋", "a product photo of loafers"),
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

    # 若有透明背景，優先只保留衣物像素
    if np.any(alpha < 250):
        mask = alpha > 20
        pixels = rgb[mask]
        if len(pixels) > 0:
            return pixels

    # 沒有透明背景時，退回全部像素
    return rgb


def _classify_neutral_light_color(pixels: np.ndarray) -> str | None:
    """
    專門處理白 / 米 / 灰 / 黑這類低飽和色。
    回傳：
      - 白色系 / 米色系 / 灰色系 / 黑色系
      - 若不屬於明顯中性色，回傳 None
    """
    if len(pixels) == 0:
        return None

    pixels_f = pixels.astype(np.float32) / 255.0
    rgb_max = pixels_f.max(axis=1)
    rgb_min = pixels_f.min(axis=1)
    chroma = rgb_max - rgb_min
    value = rgb_max

    # 低飽和像素：接近白灰黑米
    neutral_mask = chroma <= 0.18
    neutral_pixels = pixels_f[neutral_mask]

    if len(neutral_pixels) < max(30, int(len(pixels_f) * 0.15)):
        return None

    neutral_ratio = len(neutral_pixels) / len(pixels_f)

    # 中性色比例太低，代表不是主色
    if neutral_ratio < 0.35:
        return None

    nr = neutral_pixels[:, 0]
    ng = neutral_pixels[:, 1]
    nb = neutral_pixels[:, 2]

    nmax = neutral_pixels.max(axis=1)
    nmin = neutral_pixels.min(axis=1)
    nchroma = nmax - nmin

    mean_r = float(np.mean(nr))
    mean_g = float(np.mean(ng))
    mean_b = float(np.mean(nb))
    mean_v = float(np.mean(nmax))
    mean_chroma = float(np.mean(nchroma))

    # 亮色像素比例：避免白衣被少量印花或陰影拖走
    bright_mask = nmax >= 0.80
    mid_mask = (nmax >= 0.45) & (nmax < 0.80)
    dark_mask = nmax < 0.22

    bright_ratio = float(np.mean(bright_mask))
    mid_ratio = float(np.mean(mid_mask))
    dark_ratio = float(np.mean(dark_mask))

    # 偏暖程度：白 vs 米色常用
    rg_diff = abs(mean_r - mean_g)
    rb_diff = mean_r - mean_b
    gb_diff = mean_g - mean_b

    # 黑色
    if dark_ratio >= 0.55:
        return "黑色系"

    # 白色：
    # - 整體亮
    # - 飽和很低
    # - R/G/B 接近
    if (
        bright_ratio >= 0.45
        and mean_v >= 0.82
        and mean_chroma <= 0.06
        and rg_diff <= 0.04
        and abs(rb_diff) <= 0.05
        and abs(gb_diff) <= 0.05
    ):
        return "白色系"

    # 米色：
    # - 偏亮
    # - 低飽和
    # - R、G 明顯高於 B，帶黃暖感
    if (
        (bright_ratio >= 0.25 or mean_v >= 0.68)
        and mean_chroma <= 0.12
        and rb_diff >= 0.04
        and gb_diff >= 0.02
    ):
        return "米色系"

    # 灰色：
    # - 中亮度
    # - 低飽和
    # - RGB 接近
    if (
        (mid_ratio >= 0.35 or mean_v >= 0.35)
        and mean_chroma <= 0.08
        and rg_diff <= 0.05
        and abs(rb_diff) <= 0.06
        and abs(gb_diff) <= 0.06
    ):
        return "灰色系"

    return None


def _rgb_to_color_family_from_pixels(pixels: np.ndarray) -> str:
    """
    非中性色時的退回判斷：
    用較鮮明像素決定主色系，避免大面積灰白背景稀釋彩色衣物。
    """
    pixels_f = pixels.astype(np.float32) / 255.0
    r = pixels_f[:, 0]
    g = pixels_f[:, 1]
    b = pixels_f[:, 2]

    rgb_max = pixels_f.max(axis=1)
    rgb_min = pixels_f.min(axis=1)
    chroma = rgb_max - rgb_min

    # 優先保留較有顏色的像素
    colorful_mask = chroma >= 0.12
    colorful = pixels_f[colorful_mask]

    if len(colorful) < max(20, int(len(pixels_f) * 0.08)):
        colorful = pixels_f

    avg = np.mean(colorful, axis=0)
    rf, gf, bf = float(avg[0]), float(avg[1]), float(avg[2])

    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)

    # 再保底一次中性色
    if v >= 0.86 and s <= 0.10:
        return "白色系"
    if v <= 0.18:
        return "黑色系"
    if s <= 0.10 and 0.18 < v < 0.86:
        return "灰色系"

    # 彩色區間
    if h >= 0.95 or h < 0.04:
        return "紅色系"
    if 0.04 <= h < 0.12:
        return "棕色系"
    if 0.12 <= h < 0.18:
        return "米色系"
    if 0.18 <= h < 0.45:
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

    # 先做中性色專用判斷，這一步對白 / 米 / 灰最重要
    neutral_family = _classify_neutral_light_color(pixels)
    if neutral_family is not None:
        # 用符合中性色條件的像素覆蓋率作為簡單信心值
        pixels_f = pixels.astype(np.float32) / 255.0
        rgb_max = pixels_f.max(axis=1)
        rgb_min = pixels_f.min(axis=1)
        chroma = rgb_max - rgb_min
        neutral_ratio = float(np.mean(chroma <= 0.18))

        score = 0.65 + neutral_ratio * 0.30
        score = max(0.0, min(1.0, score))

        return {
            "zh": neutral_family,
            "score": score,
        }

    # 非中性色才走一般主色判斷
    family = _rgb_to_color_family_from_pixels(pixels)

    pixels_f = pixels.astype(np.float32) / 255.0
    rgb_max = pixels_f.max(axis=1)
    rgb_min = pixels_f.min(axis=1)
    chroma = rgb_max - rgb_min
    colorful_ratio = float(np.mean(chroma >= 0.12))

    score = 0.55 + colorful_ratio * 0.35
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
    casual_keys = {
        "t-shirt", "hoodie", "sweatshirt", "jeans", "pants", "shorts", "skirt",
        "sneakers", "sandals"
    }
    formal_keys = {
        "shirt", "dress", "coat", "heels", "loafers"
    }
    outdoor_keys = {"boots"}

    if occasion["score"] < 0.60:
        if category_key in casual_keys:
            occasion["key"] = "casual"
            occasion["zh"] = "日常休閒"
        elif category_key in formal_keys:
            occasion["key"] = "office"
            occasion["zh"] = "上班通勤"
        elif category_key in outdoor_keys:
            occasion["key"] = "outdoor"
            occasion["zh"] = "戶外活動"

    return occasion


def _normalize_season(category_key: str, season: Dict[str, Any]) -> Dict[str, Any]:
    if season["score"] < 0.60:
        if category_key in {"coat", "boots"}:
            season["key"] = "winter"
            season["zh"] = "冬季"
        elif category_key in {"shorts", "t-shirt", "sandals"}:
            season["key"] = "summer"
            season["zh"] = "夏季"
        elif category_key in {"hoodie", "sweatshirt", "sweater", "jacket", "sneakers", "loafers"}:
            season["key"] = "mild"
            season["zh"] = "春秋"
        elif category_key in {"heels"}:
            season["key"] = "all-season"
            season["zh"] = "四季皆可"

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