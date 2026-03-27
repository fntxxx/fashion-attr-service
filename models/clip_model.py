from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

DEVICE = "cpu"
MODEL_BACKEND = "marqo_fashionsiglip"
MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"


@dataclass(frozen=True)
class ClipBackendSpec:
    key: str
    model_name: str


BACKEND_SPEC = ClipBackendSpec(
    key=MODEL_BACKEND,
    model_name=MODEL_NAME,
)

_model = None
_preprocess = None
_tokenizer = None
_text_feature_cache: dict[tuple[str, ...], torch.Tensor] = {}


def _normalize_backend(model_backend: str | None = None) -> str:
    if model_backend is None:
        return MODEL_BACKEND

    backend_key = model_backend.strip().lower()
    if backend_key != MODEL_BACKEND:
        raise ValueError(f"Unsupported model backend: {backend_key}. Supported: {MODEL_BACKEND}")

    return MODEL_BACKEND


def get_clip_model(model_backend: str | None = None):
    global _model, _preprocess, _tokenizer

    _normalize_backend(model_backend)

    if _model is None:
        model, _, preprocess = create_model_and_transforms(MODEL_NAME)
        tokenizer = get_tokenizer(MODEL_NAME)

        model = model.to(DEVICE)
        model.eval()

        _model = model
        _preprocess = preprocess
        _tokenizer = tokenizer

    return _model, _preprocess, _tokenizer


def _ensure_list(labels: Iterable[str]) -> list[str]:
    if isinstance(labels, list):
        return labels
    return list(labels)


def _encode_image_and_texts(image: Image.Image, labels, model_backend: str | None = None):
    _normalize_backend(model_backend)
    model, preprocess, tokenizer = get_clip_model()

    labels = _ensure_list(labels)
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    cache_key = tuple(labels)

    if cache_key in _text_feature_cache:
        text_features = _text_feature_cache[cache_key]
    else:
        text_input = tokenizer(labels).to(DEVICE)

        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        _text_feature_cache[cache_key] = text_features

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features, text_features, labels


def score_texts(image: Image.Image, labels, model_backend: str | None = None):
    """
    回傳每個 label 的原始 cosine similarity。
    這個分數可跨 prompt 比較，不做 softmax。
    """
    image_features, text_features, labels = _encode_image_and_texts(
        image,
        labels,
        model_backend=model_backend,
    )

    similarity = (image_features @ text_features.T).squeeze(0)

    results = []
    for label, score in zip(labels, similarity.tolist()):
        results.append({
            "label": label,
            "score": float(score),
        })

    return results


def predict_topk(image: Image.Image, labels, topk=3, model_backend: str | None = None):
    """
    給同一批 labels 做內部排序用。
    這裡保留 softmax，方便 validate / route / category 的舊邏輯繼續運作。
    """
    raw_results = score_texts(image, labels, model_backend=model_backend)

    labels = [item["label"] for item in raw_results]
    raw_scores = torch.tensor([item["score"] for item in raw_results], dtype=torch.float32)
    probs = torch.softmax(raw_scores, dim=-1)

    values, indices = probs.topk(min(topk, len(labels)))

    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        results.append({
            "label": labels[index],
            "score": float(value),
        })

    return results


def predict_best(image: Image.Image, labels, model_backend: str | None = None):
    results = predict_topk(image, labels, topk=1, model_backend=model_backend)
    best = results[0]
    return best["label"], best["score"]


def encode_image_feature(image: Image.Image, model_backend: str | None = None):
    _normalize_backend(model_backend)
    model, preprocess, _ = get_clip_model()

    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features


def score_texts_with_image_feature(image_features, labels, model_backend: str | None = None):
    _normalize_backend(model_backend)
    model, _, tokenizer = get_clip_model()

    labels = _ensure_list(labels)
    cache_key = tuple(labels)

    if cache_key in _text_feature_cache:
        text_features = _text_feature_cache[cache_key]
    else:
        text_input = tokenizer(labels).to(DEVICE)

        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        _text_feature_cache[cache_key] = text_features

    similarity = (image_features @ text_features.T).squeeze(0)

    results = []
    for label, score in zip(labels, similarity.tolist()):
        results.append({
            "label": label,
            "score": float(score),
        })

    return results


def predict_topk_with_image_feature(image_features, labels, topk=3, model_backend: str | None = None):
    raw_results = score_texts_with_image_feature(
        image_features,
        labels,
        model_backend=model_backend,
    )

    labels = [item["label"] for item in raw_results]
    raw_scores = torch.tensor([item["score"] for item in raw_results], dtype=torch.float32)

    probs = torch.softmax(raw_scores, dim=-1)

    values, indices = probs.topk(min(topk, len(labels)))

    results = []
    for value, index in zip(values.tolist(), indices.tolist()):
        results.append({
            "label": labels[index],
            "score": float(value),
        })

    return results