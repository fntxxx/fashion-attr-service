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

    category, score = classify_category(img)
    color = extract_color(img)

    style = "casual"
    season = "spring_autumn"

    return {
        "category": category,
        "colorTone": color,
        "style": style,
        "season": season,
        "score": float(score)
    }