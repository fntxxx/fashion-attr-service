from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any
from collections import defaultdict

import requests

API_URL = "http://127.0.0.1:7860/predict"
DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
LABELS_FILE = Path("labels_full_template.csv")
REPORT_FILE = Path("test_attr_quality_report.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with open(LABELS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "file": row["file"].strip(),
                "group": row["group"].strip(),
                "subgroup": row.get("subgroup", "").strip(),
                "expected_category": row["expected_category"].strip(),
                "expected_color": row["expected_color"].strip(),
                "expected_occasions": [
                    x.strip()
                    for x in row["expected_occasions"].split("|")
                    if x.strip()
                ],
                "expected_seasons": [
                    x.strip()
                    for x in row["expected_seasons"].split("|")
                    if x.strip()
                ],
            })

    return rows


def find_image_file(filename: str) -> Path | None:
    base = Path(filename).stem

    for path in DATASET_DIR.rglob("*"):
        if path.is_file():
            if path.stem == base and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path

    return None


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def post_image(image_path: Path) -> dict[str, Any]:
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, get_mime_type(image_path))}
        response = requests.post(API_URL, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


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


def evaluate_one(label_row: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_path = find_image_file(label_row["file"])
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
        result = post_image(image_path)
    except Exception as e:
        elapsed_sec = time.perf_counter() - started_at
        return {
            "file": label_row["file"],
            "group": label_row["group"],
            "subgroup": label_row.get("subgroup", ""),
            "error": f"request_failed: {e}",
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

    predicted_category = (
        result.get("categorySelection", {}).get("selected")
        or ""
    )
    predicted_colors = result.get("colors", {}).get("selected", []) or []
    predicted_occasions = result.get("occasions", {}).get("selected", []) or []
    predicted_seasons = result.get("seasons", {}).get("selected", []) or []

    category_pass = predicted_category == label_row["expected_category"]
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
    category_rows = [
        r for r in items
        if has_expected_value(r["expected"]["category"])
    ]
    color_rows = [
        r for r in items
        if has_expected_value(r["expected"]["color"])
    ]
    occasion_rows = [
        r for r in items
        if has_expected_value(r["expected"]["occasions"])
    ]
    season_rows = [
        r for r in items
        if has_expected_value(r["expected"]["seasons"])
    ]

    return {
        "rows": len(items),
        "category_accuracy": safe_ratio(
            sum(1 for r in category_rows if r["category_pass"]),
            len(category_rows),
        ),
        "color_hit_rate": safe_ratio(
            sum(1 for r in color_rows if r["color_eval"]["hit_count"] > 0),
            len(color_rows),
        ),
        "occasion_hit_rate": safe_ratio(
            sum(1 for r in occasion_rows if r["occasion_eval"]["hit_count"] > 0),
            len(occasion_rows),
        ),
        "season_hit_rate": safe_ratio(
            sum(1 for r in season_rows if r["season_eval"]["hit_count"] > 0),
            len(season_rows),
        ),
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
    all_elapsed = [
        r.get("elapsed_sec", 0.0)
        for r in results
        if isinstance(r.get("elapsed_sec"), (int, float))
    ]

    category_confusion = defaultdict(int)

    category_rows = [
        r for r in valid_results
        if has_expected_value(r["expected"]["category"])
    ]
    color_rows = [
        r for r in valid_results
        if has_expected_value(r["expected"]["color"])
    ]
    occasion_rows = [
        r for r in valid_results
        if has_expected_value(r["expected"]["occasions"])
    ]
    season_rows = [
        r for r in valid_results
        if has_expected_value(r["expected"]["seasons"])
    ]

    for r in category_rows:
        expected = r["expected"]["category"]
        predicted = r["actual"]["category"]
        if expected != predicted:
            category_confusion[(expected, predicted)] += 1

    category_confusion_report = [
        {
            "expected": expected,
            "predicted": predicted,
            "count": count,
        }
        for (expected, predicted), count in sorted(
            category_confusion.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1]),
        )
    ]

    by_group = defaultdict(list)
    by_subgroup = defaultdict(list)

    for r in valid_results:
        by_group[r["group"]].append(r)
        subgroup_key = f'{r["group"]}/{r.get("subgroup", "") or "_"}'
        by_subgroup[subgroup_key].append(r)

    group_summary = {}
    subgroup_summary = {}

    for key, items in sorted(by_group.items()):
        group_summary[key] = build_bucket_summary(items)

    for key, items in sorted(by_subgroup.items()):
        subgroup_summary[key] = build_bucket_summary(items)

    failed_items = {
        "category": [
            {
                "file": r["file"],
                "group": r["group"],
                "subgroup": r.get("subgroup", ""),
                "expected": r["expected"]["category"],
                "predicted": r["actual"]["category"],
            }
            for r in category_rows
            if not r["category_pass"]
        ],
        "color": [
            {
                "file": r["file"],
                "group": r["group"],
                "subgroup": r.get("subgroup", ""),
                "expected": r["expected"]["color"],
                "predicted": r["actual"]["colors"],
            }
            for r in color_rows
            if r["color_eval"]["hit_count"] == 0
        ],
        "occasion": [
            {
                "file": r["file"],
                "group": r["group"],
                "subgroup": r.get("subgroup", ""),
                "expected": r["expected"]["occasions"],
                "predicted": r["actual"]["occasions"],
                "main_category_key": r["actual"]["raw"].get("mainCategoryKey"),
                "fine_category_key": r["actual"]["raw"].get("categoryKey"),
            }
            for r in occasion_rows
            if r["occasion_eval"]["hit_count"] == 0
        ],
        "season": [
            {
                "file": r["file"],
                "group": r["group"],
                "subgroup": r.get("subgroup", ""),
                "expected": r["expected"]["seasons"],
                "predicted": r["actual"]["seasons"],
                "main_category_key": r["actual"]["raw"].get("mainCategoryKey"),
                "fine_category_key": r["actual"]["raw"].get("categoryKey"),
            }
            for r in season_rows
            if r["season_eval"]["hit_count"] == 0
        ],
        "errors": error_results,
    }

    return {
        "total_rows": len(results),
        "valid_evaluated_rows": len(valid_results),
        "error_rows": len(error_results),
        "category_accuracy": safe_ratio(
            sum(1 for r in category_rows if r["category_pass"]),
            len(category_rows),
        ),
        "color_hit_rate": safe_ratio(
            sum(1 for r in color_rows if r["color_eval"]["hit_count"] > 0),
            len(color_rows),
        ),
        "occasion_hit_rate": safe_ratio(
            sum(1 for r in occasion_rows if r["occasion_eval"]["hit_count"] > 0),
            len(occasion_rows),
        ),
        "season_hit_rate": safe_ratio(
            sum(1 for r in season_rows if r["season_eval"]["hit_count"] > 0),
            len(season_rows),
        ),
        "counts": {
            "category_rows": len(category_rows),
            "color_rows": len(color_rows),
            "occasion_rows": len(occasion_rows),
            "season_rows": len(season_rows),
        },
        "by_group": group_summary,
        "by_subgroup": subgroup_summary,
        "category_confusion": category_confusion_report,
        "failed_items": failed_items,
        "time": {
            "total_elapsed_sec": round(sum(all_elapsed), 4),
            "avg_elapsed_sec_per_item": round(
                safe_ratio(sum(all_elapsed), len(all_elapsed)),
                4,
            ),
            "max_elapsed_sec": round(max(all_elapsed), 4) if all_elapsed else 0.0,
            "min_elapsed_sec": round(min(all_elapsed), 4) if all_elapsed else 0.0,
        },
    }


def main():
    script_started_at = time.perf_counter()
    labels = load_labels()
    results = [evaluate_one(row) for row in labels]
    summary = summarize(results)

    script_elapsed_sec = time.perf_counter() - script_started_at

    report = {
        "api_url": API_URL,
        "dataset_dir": str(DATASET_DIR.resolve()),
        "labels_file": str(LABELS_FILE.resolve()),
        "script_elapsed_sec": round(script_elapsed_sec, 4),
        "summary": summary,
        "items": results,
    }

    output_path = REPORT_FILE
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Attr Quality Test Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== Timing ===")
    print(
        json.dumps(
            {
                "script_elapsed_sec": round(script_elapsed_sec, 4),
                "total_elapsed_sec": summary["time"]["total_elapsed_sec"],
                "avg_elapsed_sec_per_item": summary["time"]["avg_elapsed_sec_per_item"],
                "max_elapsed_sec": summary["time"]["max_elapsed_sec"],
                "min_elapsed_sec": summary["time"]["min_elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== Category Confusion ===")
    for item in summary["category_confusion"][:10]:
        print(f'{item["expected"]} -> {item["predicted"]} : {item["count"]}')

    print("\n=== By Group ===")
    for group, data in summary["by_group"].items():
        print(
            f"{group}: rows={data['rows']}, "
            f"category={data['category_accuracy']:.4f}, "
            f"color={data['color_hit_rate']:.4f}, "
            f"occasion={data['occasion_hit_rate']:.4f}, "
            f"season={data['season_hit_rate']:.4f}"
        )

    print("\n=== Failed Counts ===")
    print(
        json.dumps(
            {
                "category": len(summary["failed_items"]["category"]),
                "color": len(summary["failed_items"]["color"]),
                "occasion": len(summary["failed_items"]["occasion"]),
                "season": len(summary["failed_items"]["season"]),
                "errors": len(summary["failed_items"]["errors"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\nreport saved to: {output_path}")


if __name__ == "__main__":
    main()