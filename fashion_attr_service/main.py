from __future__ import annotations

import io
from typing import Union

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from fashion_attr_service.api.schemas import (
    HealthResponse,
    PredictRejectedResponse,
    PredictSuccessResponse,
    ValidationErrorResponse,
    WarmupErrorResponse,
    WarmupSuccessResponse,
)
from fashion_attr_service.services.predict_pipeline import predict_attributes, run_warmup


app = FastAPI(
    title="fashion-attr-service",
    summary="服飾屬性辨識 API，提供健康檢查、warmup 與單張圖片屬性推論。",
    description=(
        "此服務部署於 Hugging Face Spaces，Swagger UI 直接顯示於 `/`。"
        "主要用途為上傳單張服飾圖片後，回傳名稱、類別、色系、場合、季節與驗證資訊。"
        "\n\n"
        "- Swagger UI：`/`\n"
        "- OpenAPI schema：`/openapi.json`\n"
        "- 檔案上傳方式：`multipart/form-data`\n"
        "- 上傳欄位名稱：`image`"
    ),
    version="1.1.0",
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


@app.on_event("startup")
def startup_event():
    result = run_warmup()
    if not result["ok"]:
        raise RuntimeError(f"fashion-attr warmup failed: {result['error']}")


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
                            "value": WarmupSuccessResponse.model_config["json_schema_extra"]["example"],
                        },
                        "error": {
                            "summary": "Warmup failed",
                            "value": WarmupErrorResponse.model_config["json_schema_extra"]["example"],
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
        },
    },
    summary="上傳圖片進行服飾屬性推論",
    description=(
        "上傳單張圖片進行服飾屬性推論。"
        "請使用 `multipart/form-data`，並以 `image` 作為欄位名稱。"
        "\n\n"
        "成功時 `ok: true`，回傳名稱、類別、色系、場合、季節與驗證資訊。"
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
    return predict_attributes(original_img)
