import sys
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cpu"

# 你要的四組候選
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

def rank(image: Image.Image, labels, model, processor, topk=3):
    texts = [f"a photo of {t}" for t in labels]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # (num_labels,)
        probs = logits.softmax(dim=0)

    top = torch.topk(probs, k=min(topk, len(labels)))
    results = [(labels[i], float(top.values[idx])) for idx, i in enumerate(top.indices)]
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_clip.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def top1(image, labels, zh_map, model, processor):
        r = rank(image, labels, model, processor, topk=1)[0]
        label, p = r[0], r[1]
        zh = zh_map.get(label, label)
        return {"en": label, "zh": zh, "score": p}

    result_category = top1(image, CATEGORY, CATEGORY_ZH, model, processor)
    result_occasion = top1(image, OCCASION, OCCASION_ZH, model, processor)
    result_color = top1(image, COLOR_TONE, COLOR_TONE_ZH, model, processor)
    result_season = top1(image, SEASON, SEASON_ZH, model, processor)

    def pct(x): return f"{x*100:.0f}%"

    print("== DEMO RESULT ==")
    print(f"類別：{result_category['zh']} ({pct(result_category['score'])})")
    print(f"場合：{result_occasion['zh']} ({pct(result_occasion['score'])})")
    print(f"色系：{result_color['zh']} ({pct(result_color['score'])})")
    print(f"季節：{result_season['zh']} ({pct(result_season['score'])})")

if __name__ == "__main__":
    main()