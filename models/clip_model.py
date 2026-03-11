import torch
import open_clip
from PIL import Image

DEVICE = "cpu"

_model = None
_preprocess = None
_tokenizer = None


def get_clip_model():
    global _model, _preprocess, _tokenizer

    if _model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        model = model.to(DEVICE)
        model.eval()

        _model = model
        _preprocess = preprocess
        _tokenizer = tokenizer

    return _model, _preprocess, _tokenizer


def _encode_image_and_texts(image: Image.Image, labels):
    model, preprocess, tokenizer = get_clip_model()

    if not isinstance(labels, list):
        labels = list(labels)

    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    text_input = tokenizer(labels).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features, labels


def score_texts(image: Image.Image, labels):
    """
    回傳每個 label 的原始 cosine similarity。
    這個分數可跨 prompt 比較，不做 softmax。
    """
    image_features, text_features, labels = _encode_image_and_texts(image, labels)

    similarity = (image_features @ text_features.T).squeeze(0)

    results = []
    for label, score in zip(labels, similarity.tolist()):
        results.append({
            "label": label,
            "score": float(score)
        })

    return results


def predict_topk(image: Image.Image, labels, topk=3):
    """
    給同一批 labels 做內部排序用。
    這裡保留 softmax，方便你原本 validate / route / category 的舊邏輯繼續運作。
    """
    raw_results = score_texts(image, labels)

    labels = [item["label"] for item in raw_results]
    raw_scores = torch.tensor([item["score"] for item in raw_results], dtype=torch.float32)
    probs = torch.softmax(raw_scores, dim=-1)

    values, indices = probs.topk(min(topk, len(labels)))

    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        results.append({
            "label": labels[index],
            "score": float(value)
        })

    return results


def predict_best(image: Image.Image, labels):
    results = predict_topk(image, labels, topk=1)
    best = results[0]
    return best["label"], best["score"]