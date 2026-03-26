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
