---
title: fashion-attr-service
emoji: 👕
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# fashion-attr-service

服飾屬性辨識服務（Fashion Attribute Service），基於 FastAPI + Marqo FashionSigLIP，部署於 Hugging Face Spaces。

提供單張服飾圖片辨識 API。成功時統一回傳 `ok + data` JSON envelope，失敗時統一回傳 `ok + error` JSON envelope，並將驗證、warmup 與推論流程收斂到伺服器端。錯誤狀態碼會依認證、請求格式、圖片內容、商品圖驗證與伺服器內部失敗分流。

---

## 🔗 文件與入口

服務啟動後：

- Swagger UI（API 文件）：`/`
- OpenAPI schema：`/openapi.json`

👉 在 Hugging Face Spaces 首頁即為 Swagger UI

---

## 🚀 本機啟動

### 1. 建立環境並安裝套件

```bash
cd /d/Projects/node/fashion-attr-service
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. 設定共用 API Token

```bash
export INTERNAL_API_TOKEN="replace-with-shared-token"
```

### 3. 啟動服務

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. 開啟

- Swagger UI：http://localhost:7860/
- OpenAPI schema：http://localhost:7860/openapi.json
- Health：http://localhost:7860/health

---

## 🐳 Docker 啟動

```bash
docker build -t fashion-attr-service .
docker run --rm -p 7860:7860 -e INTERNAL_API_TOKEN=replace-with-shared-token fashion-attr-service
```

---

## 🗂️ 專案結構

```text
app.py                  # 部署入口 thin wrapper
fashion_attr_service/
  api/                  # 認證、回應格式、schema 與 API 常數
  core/                 # 設定讀取等核心模組
  models/               # Marqo FashionSigLIP 與 YOLO 載入
  services/             # 驗證、分類、顏色、場合、季節與主推論流程
  utils/                # 類別、色系與分數輔助工具
scripts/
  debug/                # 本機除錯腳本
  deploy/               # 部署前預載資產腳本
  eval/                 # 評估相關腳本
  labels/               # 標籤檢查與產生腳本
test/                   # 正式 API contract 與不變條件測試
```

> `artifacts/weights/` 會在 Docker build 階段由 `scripts/deploy/preload_runtime_assets.py` 自動建立並預載 YOLO 權重，非目前版本控制中的主要原始碼目錄。

## 📌 API 一覽

### 1. Health

```http
GET /health
```

回傳服務名稱與目前主要 API 路徑。

### 成功回應

```json
{
  "ok": true,
  "data": {
    "service": "fashion-attr-service",
    "endpoints": {
      "warmup": "/warmup",
      "predict": "POST /predict"
    }
  }
}
```

---

### 2. Warmup

```http
GET /warmup
```

預先初始化驗證、分類與顏色流程，用來確認模型與主要推論路徑可正常啟動。

### Request

- Header：`Authorization: Bearer <INTERNAL_API_TOKEN>`

### 成功回應

```json
{
  "ok": true,
  "data": {
    "service": "fashion-attr-service",
    "model": {
      "backend": "marqo_fashionsiglip",
      "model_name": "hf-hub:Marqo/marqo-fashionSigLIP"
    },
    "warmup": {
      "validation_best_label": "t-shirt",
      "coarse_type": "upper_body",
      "category": "top",
      "color": "gray"
    }
  }
}
```

---

### 3. Predict（核心 API）

```http
POST /predict
```

### Request

- Header：`Authorization: Bearer <INTERNAL_API_TOKEN>`
- Content-Type：`multipart/form-data`
- 欄位：

| 名稱 | 型別 | 說明 |
|------|------|------|
| image | file | 要辨識的單張服飾圖片 |

### 請求規則

- `/predict` 只接受 `multipart/form-data`。
- 上傳欄位名稱固定為 `image`。
- 不提供 JSON、base64 或多檔案變體。
- 圖片必須可被 Pillow 解碼。
- 若圖片未通過服飾商品圖驗證，會以業務拒絕錯誤回傳。

### curl 範例

```bash
curl -X POST \
  'http://localhost:7860/predict' \
  -H 'Authorization: Bearer '$INTERNAL_API_TOKEN \
  -F 'image=@./sample.png;type=image/png'
```

---

## ✅ 成功回應

- Status：`200 OK`
- Content-Type：`application/json`
- Body：統一 `ok + data` envelope

```json
{
  "ok": true,
  "data": {
    "route": "product",
    "coarseType": "top",
    "name": "T 恤",
    "category": "top",
    "categoryLabel": "上衣",
    "color": "white",
    "colorLabel": "淺米白",
    "occasion": ["campusCasual", "socialGathering"],
    "season": ["spring", "summer"],
    "score": 0.91,
    "scores": {
      "mainCategory": 0.94,
      "category": 0.91,
      "occasion": 0.88,
      "color": 0.92,
      "season": 0.82
    },
    "candidates": {
      "category": [
        {"value": "top", "label": "上衣", "score": 0.91}
      ],
      "color": [
        {"value": "white", "label": "淺米白", "score": 0.92}
      ],
      "occasion": [
        {"value": "campusCasual", "label": "校園休閒", "score": 0.88}
      ],
      "season": [
        {"value": "spring", "label": "春季", "score": 0.82}
      ]
    },
    "detected": false,
    "detectedLabel": null,
    "bbox": null,
    "validation": {
      "best_label": "t-shirt",
      "valid_score": 0.93,
      "invalid_score": 0.07
    }
  }
}
```

---

## ❌ 錯誤回應 contract

所有錯誤統一為：

```json
{
  "ok": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "可讀訊息",
    "details": {}
  }
}
```

### 400 業務拒絕（非服飾商品圖）

```json
{
  "ok": false,
  "error": {
    "code": "predict_rejected",
    "message": "輸入圖片未通過服飾商品圖驗證。",
    "details": {
      "reason": "not_fashion_image",
      "validation": {
        "best_label": "person",
        "valid_score": 0.18,
        "invalid_score": 0.82
      }
    }
  }
}
```

### 401 缺少或無效的 API Token

```json
{
  "ok": false,
  "error": {
    "code": "unauthorized",
    "message": "缺少或無效的 API Token。",
    "details": {
      "reason": "missing_authorization_header"
    }
  }
}
```

### 422 請求驗證錯誤 / 圖片內容不可處理

```json
{
  "ok": false,
  "error": {
    "code": "request_validation_error",
    "message": "請求參數驗證失敗。",
    "details": {
      "fields": [
        {
          "loc": ["body", "image"],
          "msg": "Field required",
          "type": "missing"
        }
      ]
    }
  }
}
```

### 500 Warmup 失敗

```json
{
  "ok": false,
  "error": {
    "code": "warmup_failed",
    "message": "模型 warmup 失敗。",
    "details": {
      "service": "fashion-attr-service",
      "model": {
        "backend": "marqo_fashionsiglip",
        "model_name": "hf-hub:Marqo/marqo-fashionSigLIP"
      },
      "reason": "model initialization failed"
    }
  }
}
```

### 500 未分類伺服器錯誤

```json
{
  "ok": false,
  "error": {
    "code": "internal_server_error",
    "message": "伺服器發生未預期錯誤。",
    "details": null
  }
}
```

---

### 穩定錯誤碼 contract

- `400 predict_rejected`：圖片可被解碼，但未通過服飾商品圖驗證
- `401 unauthorized`：缺少、格式錯誤或內容不正確的 Bearer Token
- `422 request_validation_error`：缺少 `image` 欄位、格式不符，或圖片內容無法被正確解碼
- `500 warmup_failed`：warmup 流程失敗
- `http_exception`：HTTPException 轉換後的統一錯誤碼，實際 HTTP status 依觸發情境而定
- `500 internal_server_error`：未分類的伺服器內部錯誤

## 🧠 設計說明

### 為什麼 Swagger UI 掛在 `/`

原因：

- Hugging Face Spaces 預設首頁為 `/`
- 直接顯示 Swagger UI 可讓 API 測試與文件同步集中在同一入口
- 不需要再額外記 `/docs`

### 為什麼 `/predict` 只保留 `multipart/form-data`

原因：

- 服務核心輸入就是單張服飾圖片，上傳檔案比 JSON / base64 更直接
- 能讓 Swagger UI 與 OpenAPI 維持單一 request contract
- 可避免不同輸入格式分流造成驗證、測試與文件長期漂移

### 為什麼成功與錯誤都統一 envelope

原因：

- 呼叫端可穩定用 `ok` 判斷成功或失敗
- 成功資料固定收斂到 `data`，錯誤資訊固定收斂到 `error`
- 對前端整合、錯誤處理與 API contract 測試都比較穩定

### 目前模型與推論路徑

- 模型後端固定為 `marqo_fashionsiglip`
- 模型名稱為 `hf-hub:Marqo/marqo-fashionSigLIP`
- runtime 不提供後端切換或 fallback path
- 服務會輸出 `category`、`color`、`occasion`、`season`

### 目前保留的推論版本脈絡

目前 README 對外保留的版本描述為 **v30**，重點是：

- 保留 v28 的 occasion category-coupled prior bias
- 保留 v29 的 season family-aware secondary selection
- 保留 v30 的 occasion bridge-category prior / coupling refinement
- 這次不是重寫整個推論流程，而是延續既有主幹並收斂公開 API contract

---

## 🔢 對外公開 enum 值

### category

- `top`
- `bottom`
- `outerwear`
- `shoes`
- `skirt`
- `dress`

### occasion

- `socialGathering`
- `campusCasual`
- `businessCasual`
- `professional`

### season

- `spring`
- `summer`
- `autumn`
- `winter`

### color

- `white`
- `black`
- `gray`
- `brown`
- `yellow`
- `orange`
- `pink`
- `green`
- `blue`
- `purple`

API 回傳與 API 文件一律以上述公開值為準；內部推論仍可保留既有細粒度 key，但不直接作為對外 contract。

---

## 📦 技術棧

- FastAPI
- Uvicorn
- Pillow
- PyTorch
- Transformers
- huggingface-hub
- open_clip_torch
- ultralytics
- Docker
- Hugging Face Spaces

---

## 📌 注意事項

- 僅支援單張圖片辨識
- `/health` 不需要 API Token；`/warmup` 與 `/predict` 需要 Bearer Token
- `/predict` 只接受 `multipart/form-data`
- multipart 欄位名稱固定為 `image`
- 圖片若無法被 Pillow 解碼，會回傳 `422 request_validation_error`
- 圖片若可解碼但不符合單件服飾商品圖規則，會回傳 `400 predict_rejected`
- Docker build 時會先預載 Marqo FashionSigLIP 與 YOLO 權重，降低啟動後第一次推論的等待成本
- `color`、`occasion`、`season` 對外輸出使用固定公開 key，不直接暴露內部細粒度 key

---

## 🧪 測試建議

```bash
pytest -q
```

建議至少確認以下情境：

- `/health` 可正常回傳服務名稱與 endpoint map
- `/warmup` 在帶正確 Bearer Token 時可成功完成
- `/predict` 只接受 `multipart/form-data`，且必填欄位為 `image`
- 未帶 token、token 格式錯誤、token 錯誤時都會回傳 `401 unauthorized`
- 圖片缺失、request body 不符、圖片內容無法解碼時會回傳 `422 request_validation_error`
- 圖片通過解碼但未通過服飾商品圖驗證時會回傳 `400 predict_rejected`

若要手動驗證 OpenAPI：

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
schema = client.get('/openapi.json').json()
predict = schema['paths']['/predict']['post']
print(predict['requestBody']['content'].keys())
print(predict['requestBody']['content']['multipart/form-data']['schema'])
PY
```

預期 `/predict` 的 request content 只有 `multipart/form-data`，且 `required` 只包含 `image`。
