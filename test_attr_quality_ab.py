from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from models.clip_model import BACKEND_SPECS, CANDIDATE_MODEL_BACKEND, LEGACY_MODEL_BACKEND
from services.predict_pipeline import predict_attributes, run_warmup

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
LABELS_FILE = Path("labels_full_template.csv")
REPORT_FILE = Path("test_attr_quality_ab_report.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_BACKENDS = [LEGACY_MODEL_BACKEND, CANDIDATE_MODEL_BACKEND]


def load_labels() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with open(LABELS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "file": row["file"].strip(),
                    "group": row["group"].strip(),
                    "subgroup": row.get("subgroup", "").strip(),
                    "expected_category": row["expected_category"].strip(),
                    "expected_color": row["expected_color"].strip(),
                    "expected_occasions": [x.strip() for x in row["expected_occasions"].split("|") if x.strip()],
                    "expected_seasons": [x.strip() for x in row["expected_seasons"].split("|") if x.strip()],
                }
            )

    return rows


def find_image_file(dataset_dir: Path, filename: str) -> Path | None:
    base = Path(filename).stem

    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.stem == base and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path

    return None


def score_multiselect(predicted: list[str], expected: list[str]) -> dict[str, Any]:
    pred_set = set(predicted)
    exp_set = set(expected)
    hit_count = len(pred_set & exp_set)
    exact_match = pred_set == exp_set
    recall = hit_count / len(exp_set) if exp_set else 0.0

    return {
        "predicted": sorted(pred_set),
        "expected": sorted(exp_set),
        "hit_count": hit_count,
        "exact_match": exact_match,
        "recall": recall,
    }


def has_expected_value(value: str | list[str]) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def extract_topk_category_values(result: dict[str, Any], topk: int) -> list[str]:
    candidates = result.get("candidates", {}).get("category", []) or []
    return [str(item.get("value")) for item in candidates[:topk] if item.get("value")]


def evaluate_one(label_row: dict[str, Any], dataset_dir: Path, model_backend: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_path = find_image_file(dataset_dir, label_row["file"])
    if image_path is None:
        elapsed_sec = time.perf_counter() - started_at
        return {
            "file": label_row["file"],
            "group": label_row["group"],
            "subgroup": label_row.get("subgroup", ""),
            "error": "file_not_found",
            "passed": False,
            "elapsed_sec": round(elapsed_sec, 4),
        }

    try:
        image = load_image(image_path)
        result = predict_attributes(image, model_backend=model_backend)
    except Exception as e:
        elapsed_sec = time.perf_counter() - started_at
        return {
            "file": label_row["file"],
            "group": label_row["group"],
            "subgroup": label_row.get("subgroup", ""),
            "error": f"predict_failed: {e}",
            "passed": False,
            "elapsed_sec": round(elapsed_sec, 4),
        }

    if not result.get("ok"):
        elapsed_sec = time.perf_counter() - started_at
        return {
            "file": label_row["file"],
            "group": label_row["group"],
            "subgroup": label_row.get("subgroup", ""),
            "error": "predict_not_ok",
            "actual": result,
            "passed": False,
            "elapsed_sec": round(elapsed_sec, 4),
        }

    predicted_category = str(result.get("category") or "")
    predicted_color = str(result.get("color") or "")
    predicted_colors = [predicted_color] if predicted_color else []
    predicted_occasions = result.get("occasion", []) or []
    predicted_seasons = result.get("season", []) or []

    category_pass = predicted_category == label_row["expected_category"]
    category_top3_hit = label_row["expected_category"] in extract_topk_category_values(result, topk=3)

    expected_color_list = [label_row["expected_color"]] if label_row["expected_color"] else []
    color_eval = score_multiselect(predicted_colors, expected_color_list)
    occasion_eval = score_multiselect(predicted_occasions, label_row["expected_occasions"])
    season_eval = score_multiselect(predicted_seasons, label_row["expected_seasons"])

    group = label_row["group"]
    if group == "category":
        passed = category_pass
    elif group == "color":
        passed = color_eval["hit_count"] > 0
    elif group == "occasion":
        passed = occasion_eval["hit_count"] > 0
    elif group == "season":
        passed = season_eval["hit_count"] > 0
    else:
        passed = True

    elapsed_sec = time.perf_counter() - started_at

    return {
        "file": label_row["file"],
        "group": label_row["group"],
        "subgroup": label_row.get("subgroup", ""),
        "passed": passed,
        "elapsed_sec": round(elapsed_sec, 4),
        "category_pass": category_pass,
        "category_top3_hit": category_top3_hit,
        "expected": {
            "category": label_row["expected_category"],
            "color": label_row["expected_color"],
            "occasions": label_row["expected_occasions"],
            "seasons": label_row["expected_seasons"],
        },
        "actual": {
            "category": predicted_category,
            "colors": predicted_colors,
            "occasions": predicted_occasions,
            "seasons": predicted_seasons,
            "raw": result,
        },
        "color_eval": color_eval,
        "occasion_eval": occasion_eval,
        "season_eval": season_eval,
    }


def build_bucket_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    category_rows = [r for r in items if has_expected_value(r["expected"]["category"])]
    color_rows = [r for r in items if has_expected_value(r["expected"]["color"])]
    occasion_rows = [r for r in items if has_expected_value(r["expected"]["occasions"])]
    season_rows = [r for r in items if has_expected_value(r["expected"]["seasons"])]

    return {
        "rows": len(items),
        "category_top1_accuracy": safe_ratio(sum(1 for r in category_rows if r["category_pass"]), len(category_rows)),
        "category_top3_hit_rate": safe_ratio(sum(1 for r in category_rows if r["category_top3_hit"]), len(category_rows)),
        "color_hit_rate": safe_ratio(sum(1 for r in color_rows if r["color_eval"]["hit_count"] > 0), len(color_rows)),
        "occasion_hit_rate": safe_ratio(sum(1 for r in occasion_rows if r["occasion_eval"]["hit_count"] > 0), len(occasion_rows)),
        "season_hit_rate": safe_ratio(sum(1 for r in season_rows if r["season_eval"]["hit_count"] > 0), len(season_rows)),
        "counts": {
            "category_rows": len(category_rows),
            "color_rows": len(color_rows),
            "occasion_rows": len(occasion_rows),
            "season_rows": len(season_rows),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    all_elapsed = [r.get("elapsed_sec", 0.0) for r in results if isinstance(r.get("elapsed_sec"), (int, float))]

    category_rows = [r for r in valid_results if has_expected_value(r["expected"]["category"])]
    color_rows = [r for r in valid_results if has_expected_value(r["expected"]["color"])]
    occasion_rows = [r for r in valid_results if has_expected_value(r["expected"]["occasions"])]
    season_rows = [r for r in valid_results if has_expected_value(r["expected"]["seasons"])]

    category_confusion = defaultdict(int)
    for r in category_rows:
        expected = r["expected"]["category"]
        predicted = r["actual"]["category"]
        if expected != predicted:
            category_confusion[(expected, predicted)] += 1

    category_confusion_report = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(category_confusion.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]

    by_group = defaultdict(list)
    by_subgroup = defaultdict(list)
    for r in valid_results:
        by_group[r["group"]].append(r)
        subgroup_key = f'{r["group"]}/{r.get("subgroup", "") or "_"}'
        by_subgroup[subgroup_key].append(r)

    group_summary = {key: build_bucket_summary(items) for key, items in sorted(by_group.items())}
    subgroup_summary = {key: build_bucket_summary(items) for key, items in sorted(by_subgroup.items())}

    return {
        "rows": len(results),
        "valid_rows": len(valid_results),
        "error_rows": len(error_results),
        "avg_elapsed_sec": round(statistics.mean(all_elapsed), 4) if all_elapsed else None,
        "p95_elapsed_sec": round(sorted(all_elapsed)[max(0, int(len(all_elapsed) * 0.95) - 1)], 4) if all_elapsed else None,
        "category_top1_accuracy": safe_ratio(sum(1 for r in category_rows if r["category_pass"]), len(category_rows)),
        "category_top3_hit_rate": safe_ratio(sum(1 for r in category_rows if r["category_top3_hit"]), len(category_rows)),
        "color_hit_rate": safe_ratio(sum(1 for r in color_rows if r["color_eval"]["hit_count"] > 0), len(color_rows)),
        "occasion_hit_rate": safe_ratio(sum(1 for r in occasion_rows if r["occasion_eval"]["hit_count"] > 0), len(occasion_rows)),
        "season_hit_rate": safe_ratio(sum(1 for r in season_rows if r["season_eval"]["hit_count"] > 0), len(season_rows)),
        "group_summary": group_summary,
        "subgroup_summary": subgroup_summary,
        "category_confusion": category_confusion_report[:50],
        "errors": error_results[:50],
    }


def build_delta_summary(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "category_top1_accuracy",
        "category_top3_hit_rate",
        "color_hit_rate",
        "occasion_hit_rate",
        "season_hit_rate",
        "avg_elapsed_sec",
        "p95_elapsed_sec",
    ]

    delta = {}
    for key in keys:
        baseline_value = baseline.get(key)
        candidate_value = candidate.get(key)
        if isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)):
            delta[key] = round(candidate_value - baseline_value, 6)
        else:
            delta[key] = None
    return delta


def collect_changed_cases(baseline_items: list[dict[str, Any]], candidate_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_map = {item["file"]: item for item in baseline_items if "error" not in item}
    candidate_map = {item["file"]: item for item in candidate_items if "error" not in item}

    changed = []
    for file_name in sorted(set(baseline_map.keys()) & set(candidate_map.keys())):
        left = baseline_map[file_name]
        right = candidate_map[file_name]
        left_cat = left["actual"]["category"]
        right_cat = right["actual"]["category"]
        left_color = left["actual"]["colors"]
        right_color = right["actual"]["colors"]
        if left_cat != right_cat or left_color != right_color:
            changed.append(
                {
                    "file": file_name,
                    "expected": left["expected"],
                    "baseline": {
                        "category": left_cat,
                        "colors": left_color,
                        "occasions": left["actual"]["occasions"],
                        "seasons": left["actual"]["seasons"],
                    },
                    "candidate": {
                        "category": right_cat,
                        "colors": right_color,
                        "occasions": right["actual"]["occasions"],
                        "seasons": right["actual"]["seasons"],
                    },
                }
            )
    return changed[:100]


def run_model(dataset_dir: Path, labels: list[dict[str, Any]], model_backend: str) -> dict[str, Any]:
    backend_spec = BACKEND_SPECS[model_backend]
    warmup = run_warmup(model_backend=model_backend)
    results = [evaluate_one(row, dataset_dir, model_backend) for row in labels]
    summary = summarize(results)

    return {
        "model": {
            "backend": backend_spec.key,
            "model_name": backend_spec.model_name,
        },
        "warmup": warmup,
        "summary": summary,
        "items": results,
    }


def main() -> int:
    dataset_dir = DATASET_DIR
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    dataset_dir = dataset_dir.resolve()

    if not dataset_dir.exists():
        print(f"[ERROR] 找不到測試集資料夾：{dataset_dir}")
        return 1

    labels = load_labels()
    if not labels:
        print("[ERROR] 找不到任何標註資料。")
        return 1

    print("=" * 80)
    print("服飾屬性辨識 A/B 評估")
    print(f"DATASET_DIR  : {dataset_dir}")
    print(f"LABELS_FILE   : {LABELS_FILE.resolve()}")
    print("MODELS        :")
    for backend in MODEL_BACKENDS:
        spec = BACKEND_SPECS[backend]
        print(f"  - {backend}: {spec.model_name}")
    print("=" * 80)

    runs = {}
    for backend in MODEL_BACKENDS:
        print(f"\n[RUN] {backend}")
        runs[backend] = run_model(dataset_dir, labels, backend)
        print(json.dumps(runs[backend]["summary"], ensure_ascii=False, indent=2))

    baseline = runs[LEGACY_MODEL_BACKEND]
    candidate = runs[CANDIDATE_MODEL_BACKEND]

    report = {
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "labels_file": str(LABELS_FILE.resolve()),
            "rows": len(labels),
        },
        "evaluation_rules": {
            "same_candidate_labels": True,
            "same_scoring_logic": True,
            "same_normalization_logic": True,
            "same_formatter_logic": True,
        },
        "baseline": {
            "model": baseline["model"],
            "summary": baseline["summary"],
        },
        "candidate": {
            "model": candidate["model"],
            "summary": candidate["summary"],
        },
        "delta": build_delta_summary(baseline["summary"], candidate["summary"]),
        "changed_cases": collect_changed_cases(baseline["items"], candidate["items"]),
    }

    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n報表已輸出：{REPORT_FILE.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
