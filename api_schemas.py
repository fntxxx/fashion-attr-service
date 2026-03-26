from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EndpointMap(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "warmup": "/warmup",
                "predict": "POST /predict",
            }
        }
    )

    warmup: str = Field(..., title="Warmup Endpoint", description="warmup API 路徑。", examples=["/warmup"])
    predict: str = Field(..., title="Predict Endpoint", description="推論 API 路徑與方法提示。", examples=["POST /predict"])


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "service": "fashion-attr-service",
                "endpoints": {
                    "warmup": "/warmup",
                    "predict": "POST /predict",
                },
            }
        }
    )

    ok: bool
    service: str
    endpoints: EndpointMap


class WarmupPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "validation_best_label": "t-shirt",
                "coarse_type": "top",
                "category": "t_shirt",
                "color": "neutral_gray",
            }
        }
    )

    validation_best_label: str
    coarse_type: str
    category: str
    color: str


class WarmupSuccessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "service": "fashion-attr-service",
                "warmup": {
                    "validation_best_label": "t-shirt",
                    "coarse_type": "top",
                    "category": "t_shirt",
                    "color": "neutral_gray",
                },
            }
        }
    )

    ok: Literal[True]
    service: str
    warmup: WarmupPayload


class WarmupErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "service": "fashion-attr-service",
                "error": "model initialization failed",
            }
        }
    )

    ok: Literal[False]
    service: str
    error: str


class ValidationPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "best_label": "t-shirt",
                "valid_score": 0.93,
                "invalid_score": 0.07,
            }
        }
    )

    best_label: str
    valid_score: float
    invalid_score: float


class PredictRejectedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "reason": "not_fashion_image",
                "validation": {
                    "best_label": "person",
                    "valid_score": 0.18,
                    "invalid_score": 0.82,
                },
            }
        }
    )

    ok: Literal[False]
    reason: Literal["not_fashion_image"]
    validation: ValidationPayload


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": "top",
                "label": "上衣",
                "score": 0.91,
            }
        }
    )

    value: str
    label: str
    score: float


class CandidateGroupPayload(BaseModel):
    category: List[ScoredCandidate]
    color: List[ScoredCandidate]
    occasion: List[ScoredCandidate]
    season: List[ScoredCandidate]


class ScorePayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mainCategory": 0.94,
                "category": 0.91,
                "occasion": 0.88,
                "color": 0.92,
                "season": 0.82,
            }
        }
    )

    mainCategory: float
    category: float
    occasion: float
    color: float
    season: float


class PredictSuccessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "route": "product",
                "coarseType": "top",
                "name": "T 恤",
                "category": "top",
                "categoryLabel": "上衣",
                "color": "light_beige",
                "colorLabel": "淺米白",
                "occasion": ["campus_casual", "social"],
                "season": ["spring", "summer"],
                "score": 0.91,
                "scores": {
                    "mainCategory": 0.94,
                    "category": 0.91,
                    "occasion": 0.88,
                    "color": 0.92,
                    "season": 0.82,
                },
                "candidates": {
                    "category": [
                        {"value": "top", "label": "上衣", "score": 0.91},
                        {"value": "outer", "label": "外套", "score": 0.18},
                    ],
                    "color": [
                        {"value": "light_beige", "label": "淺米白", "score": 0.92},
                        {"value": "neutral_gray", "label": "中性灰", "score": 0.18},
                    ],
                    "occasion": [
                        {"value": "campus_casual", "label": "校園休閒", "score": 0.88},
                        {"value": "social", "label": "社交聚會", "score": 0.38},
                    ],
                    "season": [
                        {"value": "summer", "label": "夏季", "score": 0.76},
                        {"value": "spring", "label": "春季", "score": 0.70},
                    ],
                },
                "detected": False,
                "detectedLabel": None,
                "bbox": None,
                "validation": {
                    "best_label": "t-shirt",
                    "valid_score": 0.93,
                    "invalid_score": 0.07,
                },
            }
        }
    )

    ok: Literal[True]
    route: str
    coarseType: str
    name: str = Field(description="辨識出的衣物名稱，優先使用細類標籤。")
    category: str = Field(description="單選類別 key，例如 top、pants、skirt、dress、outer、shoes。")
    categoryLabel: str = Field(description="category 對應顯示文字。")
    color: str = Field(description="單選色系 key。")
    colorLabel: str = Field(description="color 對應顯示文字。")
    occasion: List[str] = Field(description="多選場合 key 陣列。")
    season: List[str] = Field(description="多選季節 key 陣列。")
    score: float
    scores: ScorePayload
    candidates: CandidateGroupPayload
    detected: bool
    detectedLabel: Optional[str] = None
    bbox: Optional[List[float]] = None
    validation: ValidationPayload


class ValidationErrorItem(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    detail: List[ValidationErrorItem]
