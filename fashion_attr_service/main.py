from __future__ import annotations

import io
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from fashion_attr_service.api.constants import (
    API_VERSION,
    ERROR_CODE_HTTP_EXCEPTION,
    ERROR_CODE_INTERNAL_SERVER,
    ERROR_CODE_REQUEST_VALIDATION,
    ERROR_CODE_WARMUP_FAILED,
    ERROR_MESSAGE_INTERNAL_SERVER,
    ERROR_MESSAGE_REQUEST_VALIDATION,
    ERROR_MESSAGE_WARMUP_FAILED,
    SERVICE_NAME,
)
from fashion_attr_service.api.exceptions import ApiErrorException
from fashion_attr_service.api.responses import build_error_response, build_success_response
from fashion_attr_service.api.schemas import (
    ApiErrorResponse,
    HealthResponse,
    PredictRejectedResponse,
    PredictSuccessResponse,
    ValidationErrorResponse,
    WarmupSuccessResponse,
)
from fashion_attr_service.services.predict_pipeline import predict_attributes, run_warmup


app = FastAPI(
    title=SERVICE_NAME,
    summary="服飾屬性辨識 API，提供健康檢查、warmup 與單張圖片屬性推論。",
    description=(
        "此服務部署於 Hugging Face Spaces，Swagger UI 直接顯示於 `/`。"
        "主要用途為上傳單張服飾圖片後，回傳名稱、類別、色系、場合、季節與驗證資訊。"
        "\n\n"
        "所有成功回傳統一為 `ok + data` envelope。"
        "所有錯誤回傳統一為 `ok + error` envelope。"
        "\n\n"
        "- Swagger UI：`/`\n"
        "- OpenAPI schema：`/openapi.json`\n"
        "- 檔案上傳方式：`multipart/form-data`\n"
        "- 上傳欄位名稱：`image`"
    ),
    version=API_VERSION,
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
def startup_event() -> None:
    result = run_warmup()
    if not result["ok"]:
        raise RuntimeError(f"fashion-attr warmup failed: {result['error']}")


@app.exception_handler(ApiErrorException)
async def handle_api_error_exception(request: Request, exc: ApiErrorException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = build_error_response(
        code=ERROR_CODE_REQUEST_VALIDATION,
        message=ERROR_MESSAGE_REQUEST_VALIDATION,
        details={"fields": exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload)


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) and exc.detail else f"HTTP {exc.status_code} 錯誤。"
    payload = build_error_response(
        code=ERROR_CODE_HTTP_EXCEPTION,
        message=message,
        details={"detail": exc.detail},
    )
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    payload = build_error_response(
        code=ERROR_CODE_INTERNAL_SERVER,
        message=ERROR_MESSAGE_INTERNAL_SERVER,
        details=None,
    )
    return JSONResponse(status_code=500, content=payload)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="取得服務健康狀態",
    description="回傳服務基本狀態、服務名稱，以及目前主要可用 API 路徑。",
    response_description="服務健康檢查結果。",
    tags=["system"],
    operation_id="getHealth",
    responses={
        200: {
            "description": "服務健康檢查成功。",
            "content": {
                "application/json": {
                    "example": HealthResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
        500: {
            "model": ApiErrorResponse,
            "description": "伺服器未預期錯誤。",
            "content": {
                "application/json": {
                    "example": ApiErrorResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
    },
)
def health() -> dict[str, Any]:
    return build_success_response(
        {
            "service": SERVICE_NAME,
            "endpoints": {
                "warmup": "/warmup",
                "predict": "POST /predict",
            },
        }
    )


@app.get(
    "/warmup",
    response_model=WarmupSuccessResponse,
    summary="執行模型 warmup",
    description=(
        "執行與服務啟動時相同的 warmup 流程。"
        "可用來確認模型、驗證流程、分類流程與顏色流程是否可正常初始化。"
    ),
    response_description="warmup 成功結果。",
    tags=["system"],
    operation_id="runWarmup",
    responses={
        200: {
            "description": "warmup 已成功完成。",
            "content": {
                "application/json": {
                    "example": WarmupSuccessResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
        500: {
            "model": ApiErrorResponse,
            "description": "warmup 執行失敗。",
            "content": {
                "application/json": {
                    "example": {
                        "ok": False,
                        "error": {
                            "code": ERROR_CODE_WARMUP_FAILED,
                            "message": ERROR_MESSAGE_WARMUP_FAILED,
                            "details": {
                                "service": SERVICE_NAME,
                                "model": {
                                    "backend": "marqo_fashionsiglip",
                                    "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                                },
                                "reason": "model initialization failed",
                            },
                        },
                    },
                }
            },
        },
    },
)
def warmup() -> dict[str, Any]:
    result = run_warmup()
    if result["ok"]:
        return result

    raise ApiErrorException(
        code=ERROR_CODE_WARMUP_FAILED,
        message=ERROR_MESSAGE_WARMUP_FAILED,
        status_code=500,
        details={
            "service": result.get("service", SERVICE_NAME),
            "model": result.get("model"),
            "reason": result.get("error"),
        },
    )


@app.post(
    "/predict",
    response_model=PredictSuccessResponse,
    responses={
        200: {
            "description": "圖片已完成服飾屬性推論。",
            "content": {
                "application/json": {
                    "example": PredictSuccessResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
        400: {
            "model": PredictRejectedResponse,
            "description": "請求圖片可被解碼，但不符合單件服飾商品圖輸入規則，例如人物穿搭照、多件商品圖或非服飾圖片。",
            "content": {
                "application/json": {
                    "example": PredictRejectedResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
        422: {
            "model": ValidationErrorResponse,
            "description": "請求格式錯誤，例如未提供 image 檔案，或 multipart/form-data 欄位名稱錯誤。",
            "content": {
                "application/json": {
                    "example": ValidationErrorResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
        500: {
            "model": ApiErrorResponse,
            "description": "伺服器未預期錯誤。",
            "content": {
                "application/json": {
                    "example": ApiErrorResponse.model_config["json_schema_extra"]["example"],
                }
            },
        },
    },
    summary="上傳圖片進行服飾屬性推論",
    description=(
        "上傳單張圖片進行服飾屬性推論。"
        "請使用 `multipart/form-data`，並以 `image` 作為欄位名稱。"
        "\n\n"
        "成功時回傳 `200 OK` 與統一的 `ok + data` envelope。"
        "若圖片可被讀取，但未通過服飾輸入驗證，回傳 `400 Bad Request` 與統一的 `ok + error` envelope。"
        "若請求格式本身錯誤，回傳 `422 Unprocessable Entity` 與統一的 `ok + error` envelope。"
    ),
    response_description="推論成功結果。輸入被拒絕與驗證失敗時皆使用統一錯誤格式。",
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
) -> dict[str, Any]:
    contents = await image.read()
    original_img = Image.open(io.BytesIO(contents)).convert("RGB")
    return predict_attributes(original_img)
