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

## Swagger UI

Open the Space homepage to use the interactive API docs:

- Root docs: `/`
- OpenAPI schema: `/openapi.json`

From the Swagger UI, you can open `POST /predict`, click **Try it out**, upload an image with the `image` field, and execute the request directly in the browser.

## Local run

Start the service locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

## Model evaluation

Use the unified single-model evaluation entry locally.

### Quality evaluation for rule / postprocess changes

Run the fixed quality test set directly against the in-process pipeline:

```bash
python test_attr_eval.py quality D:\DevData\attr_quality_testset
```

Call the local API instead:

```bash
python test_attr_eval.py quality D:\DevData\attr_quality_testset --runner api --api-url http://127.0.0.1:7860/predict
```

Only inspect specific groups or subgroups:

```bash
python test_attr_eval.py quality D:\DevData\attr_quality_testset --groups color --subgroups butter_yellow,rose_pink
```

Input validation regression check:

```bash
python test_attr_eval.py validation D:\DevData\ai_testset
```

Default label file:

- `artifacts/labels/labels_full_template.csv`

Default report output:

- `artifacts/reports/<dataset_name>_report_vN.json`

## Development scripts

- `python scripts/debug/debug_color_failures.py`
- `python scripts/labels/generate_full_labels_template.py`
- `python scripts/labels/generate_category_labels.py`
- `python scripts/labels/check_labels.py`
- `python scripts/labels/check_generated_labels.py`
- `python scripts/labels/check_missing_labels.py`
