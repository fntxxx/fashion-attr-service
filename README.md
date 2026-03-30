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

This version does **not** change the API schema. The update is in inference behavior and decision quality, mainly on multi-label occasion ranking and selection stability.

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

## Known limitations

Current known limitations of v30:

- occasion still under-predicts the second label in some dress / formal scenarios
- some bridge items can still drift between `business_casual`, `professional`, `social`, and `campus_casual`
- multi-label quality is still more limited by ranking / selection behavior than by raw signal availability in some cases
- category errors can still propagate downstream when a borderline upper-body item is classified into the wrong garment family

Typical failure patterns:

- `professional + social` or `business_casual + social` collapsing to a single label
- footwear boundary confusion on boots / heels between `social`, `business_casual`, and `professional`
- skirt-related over-expansion or under-selection on the second occasion label
- borderline cardigan / outerwear items causing downstream category-conditioned errors

## Swagger UI

Open the Space homepage to use the interactive API docs:

- Root docs: `/`
- OpenAPI schema: `/openapi.json`

From the Swagger UI, you can open `POST /predict`, click **Try it out**, upload an image with the `image` field, and execute the request directly in the browser.

## API output note

The API schema is unchanged, but consumers should treat:

- `occasion` as a ranked multi-label output
- `season` as a ranked multi-label output

Do not assume those lists are unordered.

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
