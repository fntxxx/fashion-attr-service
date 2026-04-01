from __future__ import annotations

from typing import Any, List, Literal, Optional, Union


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


class ModelInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "marqo_fashionsiglip",
                "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
            }
        }
    )

    backend: str
    model_name: str


class HealthData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "fashion-attr-service",
                "endpoints": {
                    "warmup": "/warmup",
                    "predict": "POST /predict",
                },
            }
        }
    )

    service: str
    endpoints: EndpointMap


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "data": {
                    "service": "fashion-attr-service",
                    "endpoints": {
                        "warmup": "/warmup",
                        "predict": "POST /predict",
                    },
                },
            }
        }
    )

    ok: Literal[True]
    data: HealthData


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


class WarmupData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service": "fashion-attr-service",
                "model": {
                    "backend": "marqo_fashionsiglip",
                    "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                },
                "warmup": {
                    "validation_best_label": "t-shirt",
                    "coarse_type": "top",
                    "category": "t_shirt",
                    "color": "neutral_gray",
                },
            }
        }
    )

    service: str
    model: ModelInfo
    warmup: WarmupPayload


class WarmupSuccessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "data": {
                    "service": "fashion-attr-service",
                    "model": {
                        "backend": "marqo_fashionsiglip",
                        "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                    },
                    "warmup": {
                        "validation_best_label": "t-shirt",
                        "coarse_type": "top",
                        "category": "t_shirt",
                        "color": "neutral_gray",
                    },
                },
            }
        }
    )

    ok: Literal[True]
    data: WarmupData



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


class PredictSuccessData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                        {"value": "outer", "label": "外套", "score": 0.06},
                        {"value": "dress", "label": "連身裙", "score": 0.02},
                        {"value": "pants", "label": "褲子", "score": 0.01},
                        {"value": "skirt", "label": "裙子", "score": 0.0},
                        {"value": "shoes", "label": "鞋子", "score": 0.0},
                    ],
                    "color": [
                        {"value": "light_beige", "label": "淺米白", "score": 0.92},
                        {"value": "neutral_gray", "label": "中性灰", "score": 0.06},
                    ],
                    "occasion": [
                        {"value": "campus_casual", "label": "校園休閒", "score": 0.88},
                        {"value": "social", "label": "社交聚會", "score": 0.81},
                    ],
                    "season": [
                        {"value": "spring", "label": "春季", "score": 0.82},
                        {"value": "summer", "label": "夏季", "score": 0.76},
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


class PredictSuccessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "data": PredictSuccessData.model_config["json_schema_extra"]["example"],
            }
        }
    )

    ok: Literal[True]
    data: PredictSuccessData


class ValidationErrorItem(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class RequestValidationErrorDetails(BaseModel):
    fields: List[ValidationErrorItem]


class PredictRejectedErrorDetails(BaseModel):
    reason: Literal["not_fashion_image"]
    validation: ValidationPayload


class HttpErrorDetails(BaseModel):
    detail: Any


class WarmupErrorDetails(BaseModel):
    service: str
    model: ModelInfo
    reason: str


class ApiErrorPayload(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "error": {
                    "code": "internal_server_error",
                    "message": "伺服器發生未預期錯誤。",
                    "details": None,
                },
            }
        }
    )

    ok: Literal[False]
    error: ApiErrorPayload


class PredictRejectedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "error": {
                    "code": "predict_rejected",
                    "message": "輸入圖片未通過服飾商品圖驗證。",
                    "details": {
                        "reason": "not_fashion_image",
                        "validation": {
                            "best_label": "person",
                            "valid_score": 0.18,
                            "invalid_score": 0.82,
                        },
                    },
                },
            }
        }
    )

    ok: Literal[False]
    error: ApiErrorPayload


class ValidationErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": False,
                "error": {
                    "code": "request_validation_error",
                    "message": "請求參數驗證失敗。",
                    "details": {
                        "fields": [
                            {
                                "loc": ["body", "image"],
                                "msg": "Field required",
                                "type": "missing",
                            }
                        ]
                    },
                },
            }
        }
    )

    ok: Literal[False]
    error: ApiErrorPayload
