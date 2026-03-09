from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

# 候選清單（英文）
CATEGORY = [
    "t-shirt", "shirt", "hoodie", "sweater", "jacket", "coat",
    "dress", "skirt", "pants", "jeans", "shorts"
]

OCCASION = [
    "casual outfit",
    "work outfit",
    "formal outfit",
    "sport outfit",
    "outdoor outfit",
    "party outfit",
]

COLOR_TONE = [
    "black tone clothing",
    "white tone clothing",
    "gray tone clothing",
    "blue tone clothing",
    "green tone clothing",
    "red tone clothing",
    "brown tone clothing",
    "beige tone clothing",
]

SEASON = [
    "spring outfit",
    "summer outfit",
    "autumn outfit",
    "winter outfit",
]

# 中文顯示對照
CATEGORY_ZH = {
    "t-shirt": "T 恤",
    "shirt": "襯衫",
    "hoodie": "帽T",
    "sweater": "毛衣",
    "jacket": "外套",
    "coat": "大衣",
    "dress": "洋裝",
    "skirt": "裙子",
    "pants": "長褲",
    "jeans": "牛仔褲",
    "shorts": "短褲",
}

OCCASION_ZH = {
    "casual outfit": "日常休閒",
    "work outfit": "上班通勤",
    "formal outfit": "正式場合",
    "sport outfit": "運動",
    "outdoor outfit": "戶外活動",
    "party outfit": "聚會",
}

COLOR_TONE_ZH = {
    "black tone clothing": "黑色系",
    "white tone clothing": "白色系",
    "gray tone clothing": "灰色系",
    "blue tone clothing": "藍色系",
    "green tone clothing": "綠色系",
    "red tone clothing": "紅色系",
    "brown tone clothing": "棕色系",
    "beige tone clothing": "米色系",
}

SEASON_ZH = {
    "spring outfit": "春",
    "summer outfit": "夏",
    "autumn outfit": "秋",
    "winter outfit": "冬",
}


def _rank_top1(image: Image.Image, labels, zh_map, model: CLIPModel, processor: CLIPProcessor):
    texts = [f"a photo of {t}" for t in labels]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)

    idx = int(torch.argmax(probs).item())
    score = float(probs[idx].item())
    en = labels[idx]
    zh = zh_map.get(en, en)

    return {"en": en, "zh": zh, "score": score}


app = FastAPI()

@app.on_event("startup")
def load_clip_model():
    app.state.model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    app.state.processor = CLIPProcessor.from_pretrained(MODEL_ID)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "fashion-attr-service",
        "ok": True,
        "endpoints": ["/health", "/predict"],
    }

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    model = app.state.model
    processor = app.state.processor

    category = _rank_top1(image, CATEGORY, CATEGORY_ZH, model, processor)
    occasion = _rank_top1(image, OCCASION, OCCASION_ZH, model, processor)
    colorTone = _rank_top1(image, COLOR_TONE, COLOR_TONE_ZH, model, processor)
    season = _rank_top1(image, SEASON, SEASON_ZH, model, processor)

    return {
        "category": category["zh"],
        "occasion": occasion["zh"],
        "colorTone": colorTone["zh"],
        "season": season["zh"],
        "scores": {
            "category": category["score"],
            "occasion": occasion["score"],
            "colorTone": colorTone["score"],
            "season": season["score"],
        },
        "received": {
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(raw),
        },
    }