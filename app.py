from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from services.classify_category import classify_category
from services.extract_color import extract_color

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    category_result = classify_category(img)
    color = extract_color(img)

    return {
        "mainCategory": category_result["mainCategory"],
        "mainCategoryKey": category_result["mainCategoryKey"],
        "category": category_result["category"],
        "categoryKey": category_result["categoryKey"],
        "colorTone": color,
        "style": category_result["style"],
        "season": category_result["season"],
        "scores": {
            "mainCategory": category_result["scores"]["mainCategory"],
            "category": category_result["scores"]["category"],
            "occasion": category_result["score"],
            "colorTone": category_result["score"],
            "season": category_result["score"]
        },
        "score": category_result["score"]
    }