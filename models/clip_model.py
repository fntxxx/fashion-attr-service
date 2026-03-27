from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import open_clip
import torch
from PIL import Image

DEVICE = os.getenv("FASHION_ATTR_DEVICE", "cpu")

DEFAULT_MODEL_BACKEND = os.getenv("FASHION_ATTR_MODEL_BACKEND", "marqo_fashionsiglip")
LEGACY_MODEL_BACKEND = "open_clip_vit_b32"
CANDIDATE_MODEL_BACKEND = "marqo_fashionsiglip"


@dataclass(frozen=True)
class ClipBackendSpec:
    key: str
    model_name: str
    pretrained: str | None


BACKEND_SPECS: dict[str, ClipBackendSpec] = {
    LEGACY_MODEL_BACKEND: ClipBackendSpec(
        key=LEGACY_MODEL_BACKEND,
        model_name="ViT-B-32",
        pretrained="openai",
    ),
    CANDIDATE_MODEL_BACKEND: ClipBackendSpec(
        key=CANDIDATE_MODEL_BACKEND,
        model_name="hf-hub:Marqo/marqo-fashionSigLIP",
        pretrained=None,
    ),
}

_model_cache: dict[str, torch.nn.Module] = {}
_preprocess_cache: dict[str, object] = {}
_tokenizer_cache: dict[str, object] = {}
_text_feature_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}


def resolve_backend(model_backend: str | None = None) -> ClipBackendSpec:
    backend_key = (model_backend or DEFAULT_MODEL_BACKEND).strip().lower()
    if backend_key not in BACKEND_SPECS:
        supported = ", ".join(sorted(BACKEND_SPECS.keys()))
        raise ValueError(f"Unsupported model backend: {backend_key}. Supported: {supported}")
    return BACKEND_SPECS[backend_key]



def get_clip_model(model_backend: str | None = None):
    backend = resolve_backend(model_backend)

    if backend.key not in _model_cache:
        if backend.pretrained:
            model, _, preprocess = open_clip.create_model_and_transforms(
                backend.model_name,
                pretrained=backend.pretrained,
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                backend.model_name,
            )

        tokenizer = open_clip.get_tokenizer(backend.model_name)

        model = model.to(DEVICE)
        model.eval()

        _model_cache[backend.key] = model
        _preprocess_cache[backend.key] = preprocess
        _tokenizer_cache[backend.key] = tokenizer

    return (
        _model_cache[backend.key],
        _preprocess_cache[backend.key],
        _tokenizer_cache[backend.key],
    )



def _ensure_list(labels: Iterable[str]) -> list[str]:
    if isinstance(labels, list):
        return labels
    return list(labels)



def _encode_image_and_texts(image: Image.Image, labels, model_backend: str | None = None):
    backend = resolve_backend(model_backend)
    model, preprocess, tokenizer = get_clip_model(backend.key)

    labels = _ensure_list(labels)
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    cache_key = (backend.key, tuple(labels))

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
    backend = resolve_backend(model_backend)
    model, preprocess, _ = get_clip_model(backend.key)

    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features



def score_texts_with_image_feature(image_features, labels, model_backend: str | None = None):
    backend = resolve_backend(model_backend)
    model, _, tokenizer = get_clip_model(backend.key)

    labels = _ensure_list(labels)
    cache_key = (backend.key, tuple(labels))

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
