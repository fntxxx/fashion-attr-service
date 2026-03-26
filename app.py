from typing import Any, Literal, Optional, Union, List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import io

from models.clip_model import encode_image_feature, get_clip_model
from services.classify_category import classify_category
from services.extract_color import extract_color
from services.infer_meta import (
    infer_style,
    infer_season,
    infer_occasions,
    infer_seasons,
)
from services.postprocess_category import postprocess_category
from services.validate_input import (
    validate_fashion_input,
    detect_coarse_fashion_type,
)
from utils.color_tags import build_color_payload


app = FastAPI(
    title="fashion-attr-service",
    summary="服飾屬性辨識 API，提供健康檢查、warmup 與單張圖片屬性推論。",
    description=(
        "此服務部署於 Hugging Face Spaces，Swagger UI 直接顯示於 `/`。"
        "主要用途為上傳單張服飾圖片後，回傳分類、顏色、場合、季節與驗證資訊。"
        "\n\n"
        "- Swagger UI：`/`\n"
        "- OpenAPI schema：`/openapi.json`\n"
        "- 檔案上傳方式：`multipart/form-data`\n"
        "- 上傳欄位名稱：`image`"
    ),
    version="1.0.0",
    docs_url="/",
    redoc_url=None,
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "system",
            "description": "服務狀態與模型暖機相關端點。",
        },
        {
            "name": "prediction",
            "description": "服飾圖片屬性推論端點。",
        },
    ],
)


# =========================
# Response Models
# =========================

class EndpointMap(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "warmup": "/warmup",
                "predict": "POST /predict",
            }
        }
    )

    warmup: str = Field(
        ...,
        title="Warmup Endpoint",
        description="warmup API 路徑，用於初始化模型與主要推論前置流程。",
        example="/warmup",
    )
    predict: str = Field(
        ...,
        title="Predict Endpoint",
        description="推論 API 路徑與方法提示。實際呼叫時請使用 POST 並帶入 image 檔案。",
        example="POST /predict",
    )


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

    ok: bool = Field(
        ...,
        title="Service Healthy",
        description="服務是否正常可用。此端點固定預期回傳 true。",
        example=True,
    )
    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="fashion-attr-service",
    )
    endpoints: EndpointMap = Field(
        ...,
        title="Endpoint Map",
        description="主要可用 API 路徑清單。",
    )


class WarmupPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "validation_best_label": "t-shirt",
                "coarse_type": "top",
                "category": "t_shirt",
                "color_tone": "neutral",
            }
        }
    )

    validation_best_label: str = Field(
        ...,
        title="Validation Best Label",
        description="warmup dummy 圖在輸入驗證步驟中的最佳標籤。用於確認驗證模型已成功初始化。",
        example="t-shirt",
    )
    coarse_type: str = Field(
        ...,
        title="Coarse Type",
        description="粗分類結果，表示模型推論出的服飾大方向類型。",
        example="top",
    )
    category: str = Field(
        ...,
        title="Category Key",
        description="服飾細分類 key。此欄位對應實際 payload 中的 categoryKey 類型語意。",
        example="t_shirt",
    )
    color_tone: str = Field(
        ...,
        title="Color Tone",
        description="顏色主色調結果，例如 neutral、dark、light。實際可出現值依模型輸出為準。",
        example="neutral",
    )


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
                    "color_tone": "neutral",
                },
            }
        }
    )

    ok: Literal[True] = Field(
        ...,
        title="Warmup Success",
        description="固定為 true，表示 warmup 成功完成。",
        example=True,
    )
    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="fashion-attr-service",
    )
    warmup: WarmupPayload = Field(
        ...,
        title="Warmup Result",
        description="warmup 路徑執行結果摘要。",
    )


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

    ok: Literal[False] = Field(
        ...,
        title="Warmup Failed",
        description="固定為 false，表示 warmup 執行失敗。",
        example=False,
    )
    service: str = Field(
        ...,
        title="Service Name",
        description="目前服務名稱。",
        example="fashion-attr-service",
    )
    error: str = Field(
        ...,
        title="Error Message",
        description="warmup 失敗原因。此訊息來自例外內容，主要供除錯與部署檢查使用。",
        example="model initialization failed",
    )


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

    best_label: str = Field(
        ...,
        title="Best Validation Label",
        description="輸入驗證模型認為最接近的標籤。可用於理解模型目前把圖片視為哪一類物件。",
        example="t-shirt",
    )
    valid_score: float = Field(
        ...,
        title="Valid Score",
        description="圖片屬於可接受服飾輸入的分數。數值越高代表越傾向被視為有效服飾圖片。",
        example=0.93,
    )
    invalid_score: float = Field(
        ...,
        title="Invalid Score",
        description="圖片屬於非服飾或不適合推論輸入的分數。數值越高代表越可能被拒絕。",
        example=0.07,
    )


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

    ok: Literal[False] = Field(
        ...,
        title="Prediction Rejected",
        description="固定為 false，表示請求成功送達，但圖片未通過服飾輸入驗證。",
        example=False,
    )
    reason: Literal["not_fashion_image"] = Field(
        ...,
        title="Reject Reason",
        description="固定值 `not_fashion_image`，代表此圖片被判定不是可進行服飾屬性推論的有效輸入。",
        example="not_fashion_image",
    )
    validation: ValidationPayload = Field(
        ...,
        title="Validation Detail",
        description="拒絕時仍會提供驗證分數，協助前端或測試人員理解拒絕原因。",
    )


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": "casual",
                "label": "休閒",
                "score": 0.88,
            }
        }
    )

    value: str = Field(
        ...,
        title="Candidate Value",
        description="候選值的機器可讀 key，前端整合時應優先依賴此值。",
        example="casual",
    )
    label: str = Field(
        ...,
        title="Candidate Label",
        description="候選值的顯示文字。",
        example="休閒",
    )
    score: float = Field(
        ...,
        title="Candidate Score",
        description="候選值分數，通常可視為排序或信心參考。數值越高代表越接近模型判定結果。",
        example=0.88,
    )


class CategorySelectionPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected": "top",
                "label": "上衣",
                "score": 0.91,
                "candidates": [
                    {"value": "top", "label": "上衣", "score": 0.91},
                    {"value": "outer", "label": "外套", "score": 0.18},
                    {"value": "pants", "label": "褲子", "score": 0.03},
                ],
            }
        }
    )

    selected: str = Field(
        ...,
        title="Selected Category",
        description="前端 UI 用的最終類別 key，例如 top、pants、skirt、dress、outer、shoes。",
        example="top",
    )
    label: str = Field(
        ...,
        title="Selected Label",
        description="selected 對應的顯示文字。",
        example="上衣",
    )
    score: float = Field(
        ...,
        title="Selected Score",
        description="selected 類別的代表分數。通常取主要分類與細分類中的較高值作為 UI 參考。",
        example=0.91,
    )
    candidates: List[ScoredCandidate] = Field(
        ...,
        title="Category Candidates",
        description="前端可用來呈現類別候選清單的陣列，已依分數由高到低排序。",
    )


class MultiSelectPayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "selected": ["casual", "daily"],
                "candidates": [
                    {"value": "casual", "label": "休閒", "score": 0.88},
                    {"value": "daily", "label": "日常", "score": 0.74},
                    {"value": "formal", "label": "正式", "score": 0.19},
                ],
                "threshold": 0.62,
                "maxSelected": 2,
            }
        }
    )

    selected: List[str] = Field(
        ...,
        title="Selected Values",
        description="最終選中的 value key 陣列。順序依服務回傳結果為準。",
        example=["casual", "daily"],
    )
    candidates: List[ScoredCandidate] = Field(
        ...,
        title="Candidate List",
        description="所有候選值與分數，用於前端顯示多選候選結果。",
    )
    threshold: float = Field(
        ...,
        title="Selection Threshold",
        description="多選判斷使用的門檻參考值。高於此值者較可能被納入 selected。",
        example=0.62,
    )
    maxSelected: int = Field(
        ...,
        title="Max Selected",
        description="此欄位理論上最多可回傳的 selected 項目數。",
        example=2,
    )


class ScorePayload(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mainCategory": 0.94,
                "category": 0.91,
                "occasion": 0.88,
                "colorTone": 0.97,
                "season": 0.82,
            }
        }
    )

    mainCategory: float = Field(
        ...,
        title="Main Category Score",
        description="主分類分數，對應 mainCategory / mainCategoryKey 的信心參考。",
        example=0.94,
    )
    category: float = Field(
        ...,
        title="Category Score",
        description="細分類分數，對應 category / categoryKey 的信心參考。",
        example=0.91,
    )
    occasion: float = Field(
        ...,
        title="Occasion Score",
        description="場合預測的最高候選分數，為 occasions.candidates 中最高分。",
        example=0.88,
    )
    colorTone: float = Field(
        ...,
        title="Color Tone Score",
        description="顏色相關預測的最高候選分數，為 colors.candidates 中最高分。",
        example=0.97,
    )
    season: float = Field(
        ...,
        title="Season Score",
        description="季節預測的最高候選分數，為 seasons.candidates 中最高分。",
        example=0.82,
    )


class PredictSuccessResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ok": True,
                "route": "product",
                "coarseType": "top",
                "mainCategory": "上身",
                "mainCategoryKey": "upper_body",
                "category": "T-shirt",
                "categoryKey": "t_shirt",
                "colorTone": "neutral",
                "colorTags": ["white", "gray"],
                "style": "casual",
                "season": "all_season",
                "categorySelection": {
                    "selected": "top",
                    "label": "上衣",
                    "score": 0.91,
                    "candidates": [
                        {"value": "top", "label": "上衣", "score": 0.91},
                        {"value": "outer", "label": "外套", "score": 0.18},
                    ],
                },
                "occasions": {
                    "selected": ["casual", "daily"],
                    "candidates": [
                        {"value": "casual", "label": "休閒", "score": 0.88},
                        {"value": "daily", "label": "日常", "score": 0.74},
                    ],
                    "threshold": 0.62,
                    "maxSelected": 2,
                },
                "seasons": {
                    "selected": ["spring", "summer"],
                    "candidates": [
                        {"value": "spring", "label": "春季", "score": 0.82},
                        {"value": "summer", "label": "夏季", "score": 0.77},
                    ],
                    "threshold": 0.58,
                    "maxSelected": 2,
                },
                "colors": {
                    "selected": ["white", "gray"],
                    "candidates": [
                        {"value": "white", "label": "白色", "score": 0.97},
                        {"value": "gray", "label": "灰色", "score": 0.51},
                    ],
                    "threshold": 0.5,
                    "maxSelected": 3,
                },
                "scores": {
                    "mainCategory": 0.94,
                    "category": 0.91,
                    "occasion": 0.88,
                    "colorTone": 0.97,
                    "season": 0.82,
                },
                "score": 0.91,
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

    ok: Literal[True] = Field(
        ...,
        title="Prediction Success",
        description="固定為 true，表示圖片通過驗證且已完成屬性推論。",
        example=True,
    )
    route: str = Field(
        ...,
        title="Inference Route",
        description="目前實際使用的推論路徑標記。此服務目前回傳 product。",
        example="product",
    )
    coarseType: str = Field(
        ...,
        title="Coarse Type",
        description="粗分類結果，用於表示服飾的大方向類型。",
        example="top",
    )

    # 舊欄位保留
    mainCategory: str = Field(
        ...,
        title="Legacy Main Category Label",
        description="舊欄位保留。主分類顯示文字，供既有前端或整合端相容使用。",
        example="上身",
    )
    mainCategoryKey: str = Field(
        ...,
        title="Legacy Main Category Key",
        description="舊欄位保留。主分類 key，請勿任意更名以免破壞既有 API contract。",
        example="upper_body",
    )
    category: str = Field(
        ...,
        title="Legacy Category Label",
        description="舊欄位保留。細分類顯示文字。",
        example="T-shirt",
    )
    categoryKey: str = Field(
        ...,
        title="Legacy Category Key",
        description="舊欄位保留。細分類 key，供既有前端或其他服務依賴。",
        example="t_shirt",
    )
    colorTone: str = Field(
        ...,
        title="Legacy Color Tone",
        description="舊欄位保留。顏色主色調結果。",
        example="neutral",
    )
    colorTags: List[str] = Field(
        ...,
        title="Legacy Color Tags",
        description="舊欄位保留。顏色標籤字串陣列，通常可用於簡易顯示或相容舊前端。",
        example=["white", "gray"],
    )
    style: str = Field(
        ...,
        title="Legacy Style",
        description="舊欄位保留。由既有規則推導出的風格結果。",
        example="casual",
    )
    season: str = Field(
        ...,
        title="Legacy Season",
        description="舊欄位保留。單一季節欄位，供相容舊端使用。",
        example="all_season",
    )

    # 新欄位
    categorySelection: CategorySelectionPayload = Field(
        ...,
        title="Category Selection",
        description="新欄位。提供適合前端 UI 呈現的最終類別與候選清單。",
    )
    occasions: MultiSelectPayload = Field(
        ...,
        title="Occasion Predictions",
        description="新欄位。場合多選預測結果。",
    )
    seasons: MultiSelectPayload = Field(
        ...,
        title="Season Predictions",
        description="新欄位。季節多選預測結果。",
    )
    colors: MultiSelectPayload = Field(
        ...,
        title="Color Predictions",
        description="新欄位。顏色多選預測結果，結構與 occasions / seasons 相同。",
    )

    scores: ScorePayload = Field(
        ...,
        title="Score Summary",
        description="各子任務的代表分數摘要，供前端或測試快速查看。",
    )
    score: float = Field(
        ...,
        title="Final Score",
        description="整體代表分數。目前主要以前述 category 分數為主，若啟用 detection 會再做加權。",
        example=0.91,
    )

    detected: bool = Field(
        ...,
        title="Detection Used",
        description="是否有使用偵測結果輔助最終判斷。",
        example=False,
    )
    detectedLabel: Optional[str] = Field(
        default=None,
        title="Detected Label",
        description="偵測到的物件標籤；若未啟用或未偵測到，會是 null。",
        example=None,
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        title="Bounding Box",
        description="偵測框座標，格式為數值陣列；若未使用偵測流程則為 null。",
        example=None,
    )

    validation: ValidationPayload = Field(
        ...,
        title="Validation Summary",
        description="輸入驗證分數摘要。即使推論成功也會一併回傳，便於前端與測試觀察。",
    )


class ValidationErrorItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "loc": ["body", "image"],
                "msg": "Field required",
                "type": "missing",
            }
        }
    )

    loc: List[Union[str, int]] = Field(
        ...,
        title="Error Location",
        description="驗證錯誤發生位置，例如 body.image 代表 multipart/form-data 中缺少 image 欄位。",
        example=["body", "image"],
    )
    msg: str = Field(
        ...,
        title="Error Message",
        description="FastAPI / Pydantic 產生的驗證錯誤訊息。",
        example="Field required",
    )
    type: str = Field(
        ...,
        title="Error Type",
        description="驗證錯誤類型代碼。",
        example="missing",
    )


class ValidationErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "loc": ["body", "image"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            }
        }
    )

    detail: List[ValidationErrorItem] = Field(
        ...,
        title="Validation Error Detail",
        description="請求格式驗證失敗的詳細項目列表。",
    )


# =========================
# Core Logic
# =========================

def run_warmup() -> dict:
    try:
        # 先確保 CLIP 模型已載入
        get_clip_model()

        # 用極小 dummy 圖走過主要路徑，讓推論前置初始化先完成
        dummy = Image.new("RGB", (224, 224), (255, 255, 255))

        image_features = encode_image_feature(dummy)

        validation = validate_fashion_input(dummy)
        coarse_info = detect_coarse_fashion_type(
            dummy,
            image_features=image_features,
        )
        category_result = classify_category(
            dummy,
            image_features=image_features,
        )
        color_tone = extract_color(dummy)
        color_payload = build_color_payload(color_tone)

        return {
            "ok": True,
            "service": "fashion-attr-service",
            "warmup": {
                "validation_best_label": validation["best_label"],
                "coarse_type": coarse_info["coarse_type"],
                "category": category_result["categoryKey"],
                "color_tone": color_payload["colorTone"],
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "service": "fashion-attr-service",
            "error": str(e),
        }


@app.on_event("startup")
def startup_event():
    result = run_warmup()
    if not result["ok"]:
        raise RuntimeError(f"fashion-attr warmup failed: {result['error']}")


CATEGORY_UI_OPTIONS = [
    ("top", "上衣"),
    ("pants", "褲子"),
    ("skirt", "裙子"),
    ("dress", "連身裙"),
    ("outer", "外套"),
    ("shoes", "鞋子"),
]

MAIN_CATEGORY_TO_UI = {
    "upper_body": "top",
    "pants": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "shoes": "shoes",
    # 目前 UI 沒有帽子，先保守 fallback 到 top
    "headwear": "top",
}

OUTER_FINE_KEYS = {
    "denim_jacket",
    "blazer",
    "coat",
    "puffer_jacket",
    "vest",
    "windbreaker",
}


def build_category_selection(category_result: dict) -> dict:
    fine_key = category_result["categoryKey"]
    main_key = category_result["mainCategoryKey"]

    if fine_key in OUTER_FINE_KEYS:
        selected = "outer"
    else:
        selected = MAIN_CATEGORY_TO_UI.get(main_key, "top")

    label_map = {value: label for value, label in CATEGORY_UI_OPTIONS}
    main_score = float(category_result["scores"]["mainCategory"])
    fine_score = float(category_result["scores"]["category"])
    selected_score = max(main_score, fine_score)

    candidates = []
    for value, label in CATEGORY_UI_OPTIONS:
        score = 0.03
        if value == selected:
            score = selected_score
        elif selected == "top" and value == "outer":
            score = 0.18 if fine_key not in OUTER_FINE_KEYS else 0.32
        elif selected == "outer" and value == "top":
            score = 0.20
        elif selected in {"skirt", "dress"} and value in {"skirt", "dress"} and value != selected:
            score = 0.14
        elif selected == "pants" and fine_key in {"wide_leg_pants", "shorts"} and value == "skirt":
            score = 0.08

        candidates.append({
            "value": value,
            "label": label,
            "score": float(score),
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return {
        "selected": selected,
        "label": label_map[selected],
        "score": float(selected_score),
        "candidates": candidates,
    }


# =========================
# Routes
# =========================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="取得服務健康狀態",
    description="回傳服務基本狀態、服務名稱，以及目前主要可用 API 路徑。",
    response_description="服務健康檢查結果。",
    tags=["system"],
    operation_id="getHealth",
)
def health():
    return {
        "ok": True,
        "service": "fashion-attr-service",
        "endpoints": {
            "warmup": "/warmup",
            "predict": "POST /predict",
        },
    }


@app.get(
    "/warmup",
    response_model=Union[WarmupSuccessResponse, WarmupErrorResponse],
    summary="執行模型 warmup",
    description=(
        "執行與服務啟動時相同的 warmup 流程。"
        "可用來確認模型、驗證流程、分類流程與顏色流程是否可正常初始化。"
    ),
    response_description="warmup 成功或失敗結果。",
    tags=["system"],
    operation_id="runWarmup",
    responses={
        200: {
            "description": "warmup 已完成，可能為成功或失敗結果。",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "Warmup success",
                            "value": {
                                "ok": True,
                                "service": "fashion-attr-service",
                                "warmup": {
                                    "validation_best_label": "t-shirt",
                                    "coarse_type": "top",
                                    "category": "t_shirt",
                                    "color_tone": "neutral",
                                },
                            },
                        },
                        "error": {
                            "summary": "Warmup failed",
                            "value": {
                                "ok": False,
                                "service": "fashion-attr-service",
                                "error": "model initialization failed",
                            },
                        },
                    }
                }
            },
        }
    },
)
def warmup():
    return run_warmup()


@app.post(
    "/predict",
    response_model=Union[PredictSuccessResponse, PredictRejectedResponse],
    responses={
        200: {
            "description": "圖片已完成處理。可能是推論成功，或因驗證失敗而回傳拒絕結果。",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "Prediction success",
                            "value": PredictSuccessResponse.model_config["json_schema_extra"]["example"],
                        },
                        "rejected": {
                            "summary": "Rejected because input is not a fashion image",
                            "value": PredictRejectedResponse.model_config["json_schema_extra"]["example"],
                        },
                    }
                }
            },
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "請求格式錯誤，例如未提供 image 檔案，或 multipart/form-data 欄位名稱錯誤。",
            "content": {
                "application/json": {
                    "examples": {
                        "missing_image": {
                            "summary": "Missing image field",
                            "value": {
                                "detail": [
                                    {
                                        "loc": ["body", "image"],
                                        "msg": "Field required",
                                        "type": "missing",
                                    }
                                ]
                            },
                        }
                    }
                }
            },
        },
    },
    summary="上傳圖片進行服飾屬性推論",
    description=(
        "上傳單張圖片進行服飾屬性推論。"
        "請使用 `multipart/form-data`，並以 `image` 作為欄位名稱。"
        "\n\n"
        "成功時 `ok: true`，回傳分類、顏色、場合、季節與驗證資訊。"
        "若圖片未通過服飾驗證，仍會回傳 200，但內容為 `ok: false` 的拒絕結果。"
    ),
    response_description="推論成功結果，或非服飾圖片的拒絕結果。",
    tags=["prediction"],
    operation_id="predictFashionAttributes",
)
async def predict(
    image: UploadFile = File(
        ...,
        description=(
            "要進行服飾屬性推論的單張圖片檔案。"
            "請使用 multipart/form-data 上傳，欄位名稱必須是 `image`。"
            "支援的實際格式依 Pillow 可讀取格式為準。"
        ),
    )
):
    contents = await image.read()
    original_img = Image.open(io.BytesIO(contents)).convert("RGB")

    validation = validate_fashion_input(original_img)

    if not validation["is_valid"]:
        return {
            "ok": False,
            "reason": "not_fashion_image",
            "validation": {
                "best_label": validation["best_label"],
                "valid_score": validation["valid_score"],
                "invalid_score": validation["invalid_score"],
            },
        }

    route = "product"
    image_features = encode_image_feature(original_img)

    coarse_info = detect_coarse_fashion_type(
        original_img,
        image_features=image_features,
    )

    detection = {
        "detected": False,
        "label": None,
        "mainCategoryKey": None,
        "bbox": None,
        "score": 0.0,
    }

    working_img = original_img

    category_result = classify_category(
        working_img,
        image_features=image_features,
    )
    color_tone = extract_color(working_img)
    color_payload = build_color_payload(color_tone)

    category_result = postprocess_category(
        category_result,
        working_img,
        color_tone=color_tone,
        route=route,
        coarse_info=coarse_info,
        validation=validation,
    )

    legacy_style = infer_style(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    legacy_season = infer_season(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )

    occasions = infer_occasions(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    seasons = infer_seasons(
        category_result["mainCategoryKey"],
        category_result["categoryKey"],
    )
    category_selection = build_category_selection(category_result)

    final_score = category_result["scores"]["category"]

    if detection["detected"]:
        final_score = (
            category_result["scores"]["category"] * 0.75
            + detection["score"] * 0.25
        )

    return {
        "ok": True,
        "route": route,
        "coarseType": coarse_info["coarse_type"],

        # 舊欄位保留
        "mainCategory": category_result["mainCategory"],
        "mainCategoryKey": category_result["mainCategoryKey"],
        "category": category_result["category"],
        "categoryKey": category_result["categoryKey"],
        "colorTone": color_payload["colorTone"],
        "colorTags": color_payload["colorTags"],
        "style": legacy_style,
        "season": legacy_season,

        # 新欄位
        "categorySelection": category_selection,
        "occasions": {
            "selected": occasions["selected"],
            "candidates": occasions["candidates"],
            "threshold": 0.62,
            "maxSelected": 2,
        },
        "seasons": {
            "selected": seasons["selected"],
            "candidates": seasons["candidates"],
            "threshold": 0.58,
            "maxSelected": 2,
        },
        "colors": color_payload["colors"],

        "scores": {
            "mainCategory": category_result["scores"]["mainCategory"],
            "category": category_result["scores"]["category"],
            "occasion": float(max([x["score"] for x in occasions["candidates"]] or [0.0])),
            "colorTone": float(max([x["score"] for x in color_payload["colors"]["candidates"]] or [0.0])),
            "season": float(max([x["score"] for x in seasons["candidates"]] or [0.0])),
        },
        "score": float(final_score),

        "detected": detection["detected"],
        "detectedLabel": detection["label"],
        "bbox": detection["bbox"],

        "validation": {
            "best_label": validation["best_label"],
            "valid_score": validation["valid_score"],
            "invalid_score": validation["invalid_score"],
        },
    }