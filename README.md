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

FastAPI service for clothing attribute prediction.

## Overview

This Hugging Face Space serves the FastAPI Swagger UI directly at the root path:

- `GET /` → Swagger UI
- `GET /openapi.json` → OpenAPI schema
- `GET /health` → basic service metadata / health response
- `GET /warmup` → run warmup flow and report warmup result
- `POST /predict` → upload an image file for clothing attribute prediction

## Version

Current deployed runtime target: **v30**

v30 is the retained production candidate after:

- keeping the v28 occasion category-coupled prior bias
- keeping the v29 season family-aware secondary selection
- keeping the v30 occasion bridge-category prior / coupling refinement

This revision does not change the core inference rules. It unifies `/predict` to a single image-upload contract and synchronizes Swagger / OpenAPI, tests, and examples with that contract.

## Predict request format

`POST /predict` only accepts `multipart/form-data`.

Field requirements:

- field name must be `image`
- content should be a single decodable image file

Example:

```bash
curl -X POST "http://localhost:7860/predict"   -F "image=@./sample.png"
```

Swagger UI notes:

- `/predict` request body now documents only `multipart/form-data`
- the upload field name is fixed to `image`
- there is no JSON or base64 request variant

## Unified API contract

All HTTP success responses now use the same envelope:

```json
{
  "ok": true,
  "data": {
    "...": "endpoint-specific payload"
  }
}
```

All HTTP error responses now use the same envelope:

```json
{
  "ok": false,
  "error": {
    "code": "machine_readable_error_code",
    "message": "human readable message",
    "details": {
      "...": "structured details"
    }
  }
}
```

Contract rules:

- success payload is always inside `data`
- error payload is always inside `error`
- `error.code` is the machine-readable discriminator
- `error.message` is the human-readable message
- `error.details` contains structured context such as validation fields or rejection details

## Runtime structure

```text
.
├─ app.py
├─ fashion_attr_service/
│  ├─ main.py
│  ├─ api/
│  ├─ models/
│  ├─ services/
│  └─ utils/
├─ scripts/
│  ├─ debug/
│  ├─ eval/
│  └─ labels/
└─ artifacts/
   ├─ labels/
   ├─ reports/
   └─ weights/
```

- `fashion_attr_service/`: API runtime 與推論流程。
- `scripts/`: 測試、除錯、標註產生等開發期腳本。
- `artifacts/`: 標註範本、報表、權重等非核心程式資產。
- `app.py`: 部署入口 thin wrapper，讓 Docker / Uvicorn 啟動方式維持不變。

## Model backend

The service uses a single image-text backbone throughout the whole inference path:

- `marqo_fashionsiglip` → `hf-hub:Marqo/marqo-fashionSigLIP`

There is no backend switch or fallback path in the service runtime.

## Inference behavior

The service predicts:

- `category` (single-label)
- `color` (single-label)
- `occasion` (multi-label)
- `season` (multi-label)

### Category

- Category is inferred through a staged category pipeline.
- Runtime output is affected by category classification plus category postprocess rules.
- Category output is also used downstream as conditioning input for attribute inference.

### Color

- Color uses the existing dedicated color extraction path.
- v30 does not change the color pipeline behavior.

### Occasion

v30 occasion behavior:

- uses category-coupled prior blending
- combines fine-category prior and main-category prior
- applies family-aware biasing inside occasion scoring
- strengthens prior influence on bridge categories such as shirt, knit, skirt, and boots
- keeps multi-label output ranking-based rather than treating labels as fully independent

Practical implication:

- v30 is more stable on `business_casual` / `professional` / `social` boundaries than earlier retained versions
- ordering matters because second-label admission still depends on ranked candidates and selection gates

### Season

- Season remains multi-label and ranking-based.
- v29 family-aware secondary selection is retained in v30.
- v30 does not introduce additional season-specific logic changes.

## Evaluation notes

The project uses strict evaluation for category / color and multi-label evaluation for occasion / season.

- `category`: primary metric = `category_top1_accuracy`
- `color`: primary metric = `color_hit_rate`
- `occasion`: **exact-match only** for pass / primary score
- `season`: exact-match, with **spring/autumn equivalence** allowed by evaluation policy

This difference matters when comparing occasion and season pass-rate numbers.

## Retained quality status for v30

Based on the retained local evaluation reports used for release gating:

- category metrics remain stable
- color metrics remain stable
- season metrics remain stable relative to v29
- occasion exact-match quality improves relative to v29

v30 should therefore be understood as an **occasion-quality refinement release**, not as a category, color, or season rewrite.
