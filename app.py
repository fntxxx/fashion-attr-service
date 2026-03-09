from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

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
# Label 設計
# =========================
# 每個 tuple: (英文 label, 中文顯示)
CATEGORY_LABELS = [
    ("t-shirt", "T 恤"),
    ("shirt", "襯衫"),
    ("hoodie", "帽T"),
    ("sweater", "毛衣"),
    ("jacket", "外套"),
    ("coat", "大衣"),
    ("dress", "洋裝"),
    ("skirt", "裙子"),
    ("jeans", "牛仔褲"),
    ("pants", "長褲"),
    ("shorts", "短褲"),
]

OCCASION_LABELS = [
    ("casual daily wear", "日常休閒"),
    ("office wear", "上班通勤"),
    ("formal occasion wear", "正式場合"),
    ("sportswear", "運動"),
    ("outdoor activity wear", "戶外活動"),
    ("party wear", "聚會"),
]

# 欄位名稱維持 colorTone，不動前端
# 但實際判斷邏輯改成主色系，會比抽象 tone 穩很多
COLOR_LABELS = [
    ("black", "黑色系"),
    ("white", "白色系"),
    ("gray", "灰色系"),
    ("blue", "藍色系"),
    ("green", "綠色系"),
    ("red", "紅色系"),
    ("brown", "棕色系"),
    ("beige", "米色系"),
]

# 不硬分春/秋，改成更可判斷的穿著氣候，再映射成前端顯示
SEASON_LABELS = [
    ("hot weather", "夏季"),
    ("mild weather", "春秋"),
    ("cold weather", "冬季"),
    ("all-season wear", "四季皆可"),
]

CATEGORY_PROMPT = "a product photo of a {}"
OCCASION_PROMPT = "this clothing item is suitable for {}"
COLOR_PROMPT = "the main color of this clothing item is {}"
SEASON_PROMPT = "this clothing item is best for {}"


# =========================
# 模型 lazy load
# =========================
def get_clip():
    if app.state.model is None or app.state.processor is None:
        app.state.model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        app.state.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        app.state.model.eval()
    return app.state.model, app.state.processor


# =========================
# 推論工具
# =========================
def _rank_top1(
    image: Image.Image,
    label_pairs,
    prompt_template: str,
    model: CLIPModel,
    processor: CLIPProcessor,
):
    en_labels = [en for en, _ in label_pairs]
    zh_map = {en: zh for en, zh in label_pairs}

    texts = [prompt_template.format(en) for en in en_labels]

    inputs = processor(
        text=texts,
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
    score = float(probs[idx].item())

    en = en_labels[idx]
    zh = zh_map[en]

    return {
        "en": en,
        "zh": zh,
        "score": score,
    }


def _open_image(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


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

    image = _open_image(raw)
    model, processor = get_clip()

    category = _rank_top1(
        image=image,
        label_pairs=CATEGORY_LABELS,
        prompt_template=CATEGORY_PROMPT,
        model=model,
        processor=processor,
    )

    occasion = _rank_top1(
        image=image,
        label_pairs=OCCASION_LABELS,
        prompt_template=OCCASION_PROMPT,
        model=model,
        processor=processor,
    )

    color_tone = _rank_top1(
        image=image,
        label_pairs=COLOR_LABELS,
        prompt_template=COLOR_PROMPT,
        model=model,
        processor=processor,
    )

    season = _rank_top1(
        image=image,
        label_pairs=SEASON_LABELS,
        prompt_template=SEASON_PROMPT,
        model=model,
        processor=processor,
    )

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