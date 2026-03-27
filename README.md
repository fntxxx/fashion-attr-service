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

## Model backend

The service now uses a single image-text backbone throughout the whole inference path:

- `marqo_fashionsiglip` → `hf-hub:Marqo/marqo-fashionSigLIP`

There is no backend switch or fallback path in the service runtime.

## Swagger UI

Open the Space homepage to use the interactive API docs:

- Root docs: `/`
- OpenAPI schema: `/openapi.json`

From the Swagger UI, you can open `POST /predict`, click **Try it out**, upload an image with the `image` field, and execute the request directly in the browser.

## Endpoints

### `GET /health`

Returns a basic health / service metadata payload.

### `GET /warmup`

Runs the same warmup path used during startup and returns the warmup result.

### `POST /predict`

Accepts a multipart file upload using the `image` field and returns clothing attribute prediction results.

## Predict response shape

`POST /predict` 成功時會回傳前端直接可用的欄位：

- `name`: 衣物細類名稱
- `category`: 單選類別 key
- `color`: 單選色系 key
- `occasion`: 多選場合 key 陣列
- `season`: 多選季節 key 陣列

另外會保留 `validation`、`scores`、`candidates` 等除錯與觀察欄位。

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

Input validation regression check

```bash
python test_attr_eval.py validation D:\DevData\ai_testset
```

The default report file is test_attr_eval_report.json.