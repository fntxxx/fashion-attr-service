from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image

DEFAULT_API_URL = "http://127.0.0.1:7860/predict"
DEFAULT_QUALITY_DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
DEFAULT_VALIDATION_DATASET_DIR = Path(r"D:\DevData\ai_testset")
DEFAULT_LABELS_FILE = Path("artifacts/labels/labels_full_template.csv")
DEFAULT_REPORT_FILE = Path("artifacts/reports/test_attr_eval_report.json")
DEFAULT_TIMEOUT_SECONDS = 120
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

VALIDATION_EXPECTED_RULES = {
    "valid_product": {"ok": True},
    "flatlay": {"ok": True},
    "person_outfit": {"ok": False, "reason": "not_fashion_image"},
    "multi_item": {"ok": False, "reason": "not_fashion_image"},
    "invalid": {"ok": False, "reason": "not_fashion_image"},
}


@dataclass(frozen=True)
class QualityLabelRow:
    file: str
    group: str
    subgroup: str
    expected_category: str
    expected_color: str
    expected_occasions: list[str]
    expected_seasons: list[str]


@dataclass(frozen=True)
class ValidationCase:
    category: str
    file_path: Path

    @property
    def expected(self) -> dict[str, Any]:
        return VALIDATION_EXPECTED_RULES[self.category]


class PredictRunner:
    kind: str

    def warmup(self) -> dict[str, Any] | None:
        return None

    def predict(self, image_path: Path) -> dict[str, Any]:
        raise NotImplementedError


class ApiPredictRunner(PredictRunner):
    def __init__(self, api_url: str, timeout_seconds: int) -> None:
        self.kind = "api"
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def predict(self, image_path: Path) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        with image_path.open("rb") as file_obj:
            files = {"image": (image_path.name, file_obj, mime_type)}
            response = requests.post(self.api_url, files=files, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


class PipelinePredictRunner(PredictRunner):
    def __init__(self) -> None:
        self.kind = "pipeline"
        self._predict_attributes = None
        self._run_warmup = None

    def _ensure_loaded(self) -> None:
        if self._predict_attributes is not None and self._run_warmup is not None:
            return

        from fashion_attr_service.services.predict_pipeline import predict_attributes, run_warmup

        self._predict_attributes = predict_attributes
        self._run_warmup = run_warmup

    def warmup(self) -> dict[str, Any] | None:
        self._ensure_loaded()
        assert self._run_warmup is not None
        return self._run_warmup()

    def predict(self, image_path: Path) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._predict_attributes is not None

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return self._predict_attributes(rgb_image, include_debug=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="單一入口的服飾屬性驗證腳本：支援品質驗證與輸入驗證。",
    )
    parser.add_argument(
        "task",
        choices=["quality", "validation"],
        nargs="?",
        default="quality",
        help="quality=固定測試集品質驗證；validation=輸入有效性驗證。",
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        help="測試集資料夾路徑；未提供時依 task 使用預設值。",
    )
    parser.add_argument(
        "--runner",
        choices=["pipeline", "api"],
        default="pipeline",
        help="pipeline=直接呼叫本地推論流程；api=呼叫 /predict API。",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="runner=api 時使用的 /predict URL。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="runner=api 時的請求逾時秒數。",
    )
    parser.add_argument(
        "--labels-file",
        default=str(DEFAULT_LABELS_FILE),
        help="quality 任務使用的標註 CSV。",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_REPORT_FILE),
        help="輸出的報表 JSON 路徑。",
    )
    parser.add_argument(
        "--groups",
        help="quality 任務只跑指定群組，逗號分隔，例如 category,color。",
    )
    parser.add_argument(
        "--subgroups",
        help="quality 任務只跑指定 subgroup，逗號分隔，例如 butter_yellow,rose_pink。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="quality 任務計算 category top-k 命中率的 k 值。",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="runner=pipeline 時略過 warmup。",
    )
    parser.add_argument(
        "--failures-limit",
        type=int,
        default=50,
        help="報表中保留的失敗案例上限。",
    )
    return parser


def parse_csv_list(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None
    values = {item.strip() for item in raw_value.split(",") if item.strip()}
    return values or None


def resolve_dataset_dir(task: str, dataset_dir_arg: str | None) -> Path:
    if dataset_dir_arg:
        return Path(dataset_dir_arg).resolve()
    if task == "validation":
        return DEFAULT_VALIDATION_DATASET_DIR.resolve()
    return DEFAULT_QUALITY_DATASET_DIR.resolve()


def build_runner(args: argparse.Namespace) -> PredictRunner:
    if args.runner == "api":
        return ApiPredictRunner(api_url=args.api_url, timeout_seconds=args.timeout)
    return PipelinePredictRunner()


def get_dataset_name(dataset_dir: Path) -> str:
    resolved = dataset_dir.resolve()
    return resolved.name or "dataset"


def get_next_report_version(base_dir: Path, dataset_name: str) -> int:
    version = 1
    while (base_dir / f"{dataset_name}_report_v{version}.json").exists():
        version += 1
    return version


def resolve_report_file_path(report_file_arg: str, dataset_dir: Path) -> Path:
    if report_file_arg != str(DEFAULT_REPORT_FILE):
        return Path(report_file_arg).resolve()

    dataset_name = get_dataset_name(dataset_dir)
    reports_dir = Path.cwd() / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    version = get_next_report_version(reports_dir, dataset_name)
    return reports_dir / f"{dataset_name}_report_v{version}.json"


def load_quality_labels(labels_file: Path) -> list[QualityLabelRow]:
    rows: list[QualityLabelRow] = []
    with labels_file.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                QualityLabelRow(
                    file=row["file"].strip(),
                    group=row["group"].strip(),
                    subgroup=(row.get("subgroup") or "").strip(),
                    expected_category=row["expected_category"].strip(),
                    expected_color=row["expected_color"].strip(),
                    expected_occasions=[item.strip() for item in row["expected_occasions"].split("|") if item.strip()],
                    expected_seasons=[item.strip() for item in row["expected_seasons"].split("|") if item.strip()],
                )
            )
    return rows


def filter_quality_labels(
    rows: list[QualityLabelRow],
    groups: set[str] | None,
    subgroups: set[str] | None,
) -> list[QualityLabelRow]:
    filtered = rows
    if groups is not None:
        filtered = [row for row in filtered if row.group in groups]
    if subgroups is not None:
        filtered = [row for row in filtered if row.subgroup in subgroups]
    return filtered


def find_image_file(dataset_dir: Path, filename: str) -> Path | None:
    base_name = Path(filename).stem
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.stem == base_name and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def has_expected_value(value: str | list[str]) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def score_multiselect(predicted: Iterable[str], expected: Iterable[str]) -> dict[str, Any]:
    predicted_set = {item for item in predicted if item}
    expected_set = {item for item in expected if item}
    hit_count = len(predicted_set & expected_set)
    exact_match = predicted_set == expected_set
    recall = safe_ratio(hit_count, len(expected_set)) if expected_set else 0.0
    precision = safe_ratio(hit_count, len(predicted_set)) if predicted_set else 0.0
    false_positive = sorted(predicted_set - expected_set)
    false_negative = sorted(expected_set - predicted_set)

    return {
        "predicted": sorted(predicted_set),
        "expected": sorted(expected_set),
        "hit_count": hit_count,
        "exact_match": exact_match,
        "partial_hit": hit_count > 0 and not exact_match,
        "recall": recall,
        "precision": precision,
        "predicted_count": len(predicted_set),
        "expected_count": len(expected_set),
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def summarize_multiselect_rows(items: list[dict[str, Any]], eval_key: str) -> dict[str, Any]:
    if not items:
        return {
            "pass_rate": 0.0,
            "hit_rate": 0.0,
            "exact_match_rate": 0.0,
            "partial_hit_rate": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "avg_false_positive_count": 0.0,
            "avg_false_negative_count": 0.0,
            "avg_predicted_count": 0.0,
            "avg_expected_count": 0.0,
        }

    evals = [item[eval_key] for item in items]
    exact_match_rate = safe_ratio(sum(1 for metric in evals if metric["exact_match"]), len(evals))
    return {
        "pass_rate": exact_match_rate,
        "hit_rate": safe_ratio(sum(1 for metric in evals if metric["hit_count"] > 0), len(evals)),
        "exact_match_rate": exact_match_rate,
        "partial_hit_rate": safe_ratio(sum(1 for metric in evals if metric["partial_hit"]), len(evals)),
        "avg_precision": safe_ratio(sum(metric["precision"] for metric in evals), len(evals)),
        "avg_recall": safe_ratio(sum(metric["recall"] for metric in evals), len(evals)),
        "false_positive_rate": safe_ratio(sum(1 for metric in evals if metric["false_positive"]), len(evals)),
        "false_negative_rate": safe_ratio(sum(1 for metric in evals if metric["false_negative"]), len(evals)),
        "avg_false_positive_count": safe_ratio(sum(len(metric["false_positive"]) for metric in evals), len(evals)),
        "avg_false_negative_count": safe_ratio(sum(len(metric["false_negative"]) for metric in evals), len(evals)),
        "avg_predicted_count": safe_ratio(sum(metric["predicted_count"] for metric in evals), len(evals)),
        "avg_expected_count": safe_ratio(sum(metric["expected_count"] for metric in evals), len(evals)),
    }


def extract_topk_category_values(result: dict[str, Any], topk: int) -> list[str]:
    candidates = result.get("candidates", {}).get("category", []) or []
    values: list[str] = []
    for item in candidates[:topk]:
        value = item.get("value") or item.get("label") or item.get("category")
        if value:
            values.append(str(value))
    return values


def build_quality_debug_info(result: dict[str, Any]) -> dict[str, Any]:
    debug = result.get("_debug") or {}
    return {
        "pre_postprocess_category": debug.get("pre_postprocess_category"),
        "postprocess_category": debug.get("postprocess_category"),
        "coarse_type": debug.get("coarse_type") or result.get("coarseType"),
        "coarse_score": debug.get("coarse_score"),
        "candidate_score_map": debug.get("candidate_score_map"),
    }


def build_multilabel_case_breakdown(items: list[dict[str, Any]], eval_key: str) -> dict[str, Any]:
    evals = [item[eval_key] for item in items]
    return {
        "over_predict_cases": sum(1 for metric in evals if metric["false_positive"] and not metric["false_negative"]),
        "under_predict_cases": sum(1 for metric in evals if metric["false_negative"] and not metric["false_positive"]),
        "hit_but_not_exact_cases": sum(1 for metric in evals if metric["hit_count"] > 0 and not metric["exact_match"]),
    }


def evaluate_quality_case(
    row: QualityLabelRow,
    dataset_dir: Path,
    runner: PredictRunner,
    topk: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    image_path = find_image_file(dataset_dir, row.file)
    if image_path is None:
        return {
            "file": row.file,
            "group": row.group,
            "subgroup": row.subgroup,
            "error": "file_not_found",
            "passed": False,
            "elapsed_sec": round(time.perf_counter() - started_at, 4),
        }

    try:
        result = runner.predict(image_path)
    except Exception as exc:
        return {
            "file": row.file,
            "group": row.group,
            "subgroup": row.subgroup,
            "error": f"predict_failed: {exc}",
            "passed": False,
            "elapsed_sec": round(time.perf_counter() - started_at, 4),
        }

    if not result.get("ok"):
        return {
            "file": row.file,
            "group": row.group,
            "subgroup": row.subgroup,
            "error": "predict_not_ok",
            "passed": False,
            "elapsed_sec": round(time.perf_counter() - started_at, 4),
            "actual": result,
        }

    predicted_category = str(result.get("category") or "")
    predicted_color = str(result.get("color") or "")
    predicted_colors = [predicted_color] if predicted_color else []
    predicted_occasions = [str(item) for item in (result.get("occasion", []) or [])]
    predicted_seasons = [str(item) for item in (result.get("season", []) or [])]

    category_pass = predicted_category == row.expected_category
    category_topk_values = extract_topk_category_values(result, topk=topk)
    category_topk_hit = row.expected_category in category_topk_values if row.expected_category else False

    color_eval = score_multiselect(predicted_colors, [row.expected_color] if row.expected_color else [])
    occasion_eval = score_multiselect(predicted_occasions, row.expected_occasions)
    season_eval = score_multiselect(predicted_seasons, row.expected_seasons)

    if row.group == "category":
        passed = category_pass
    elif row.group == "color":
        passed = color_eval["hit_count"] > 0
    elif row.group == "occasion":
        passed = occasion_eval["exact_match"]
    elif row.group == "season":
        passed = season_eval["exact_match"]
    else:
        passed = True

    return {
        "file": row.file,
        "group": row.group,
        "subgroup": row.subgroup,
        "passed": passed,
        "elapsed_sec": round(time.perf_counter() - started_at, 4),
        "category_pass": category_pass,
        "category_topk_hit": category_topk_hit,
        "category_topk_values": category_topk_values,
        "expected": {
            "category": row.expected_category,
            "color": row.expected_color,
            "occasions": row.expected_occasions,
            "seasons": row.expected_seasons,
        },
        "actual": {
            "category": predicted_category,
            "colors": predicted_colors,
            "occasions": predicted_occasions,
            "seasons": predicted_seasons,
            "raw": result,
        },
        "debug": build_quality_debug_info(result),
        "color_eval": color_eval,
        "occasion_eval": occasion_eval,
        "season_eval": season_eval,
    }


def build_quality_bucket_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    category_rows = [item for item in items if has_expected_value(item["expected"]["category"])]
    color_rows = [item for item in items if has_expected_value(item["expected"]["color"])]
    occasion_rows = [item for item in items if has_expected_value(item["expected"]["occasions"])]
    season_rows = [item for item in items if has_expected_value(item["expected"]["seasons"])]

    color_metrics = summarize_multiselect_rows(color_rows, "color_eval")
    occasion_metrics = summarize_multiselect_rows(occasion_rows, "occasion_eval")
    season_metrics = summarize_multiselect_rows(season_rows, "season_eval")
    occasion_breakdown = build_multilabel_case_breakdown(occasion_rows, "occasion_eval")
    season_breakdown = build_multilabel_case_breakdown(season_rows, "season_eval")

    return {
        "rows": len(items),
        "category_top1_accuracy": safe_ratio(sum(1 for item in category_rows if item["category_pass"]), len(category_rows)),
        "category_topk_hit_rate": safe_ratio(sum(1 for item in category_rows if item["category_topk_hit"]), len(category_rows)),
        "color_hit_rate": color_metrics["hit_rate"],
        "occasion_pass_rate": occasion_metrics["pass_rate"],
        "occasion_hit_rate": occasion_metrics["hit_rate"],
        "occasion_exact_match_rate": occasion_metrics["exact_match_rate"],
        "occasion_partial_hit_rate": occasion_metrics["partial_hit_rate"],
        "occasion_avg_precision": occasion_metrics["avg_precision"],
        "occasion_avg_recall": occasion_metrics["avg_recall"],
        "occasion_false_positive_rate": occasion_metrics["false_positive_rate"],
        "occasion_false_negative_rate": occasion_metrics["false_negative_rate"],
        "occasion_avg_false_positive_count": occasion_metrics["avg_false_positive_count"],
        "occasion_avg_false_negative_count": occasion_metrics["avg_false_negative_count"],
        "occasion_avg_predicted_count": occasion_metrics["avg_predicted_count"],
        "season_pass_rate": season_metrics["pass_rate"],
        "season_hit_rate": season_metrics["hit_rate"],
        "season_exact_match_rate": season_metrics["exact_match_rate"],
        "season_partial_hit_rate": season_metrics["partial_hit_rate"],
        "season_avg_precision": season_metrics["avg_precision"],
        "season_avg_recall": season_metrics["avg_recall"],
        "season_false_positive_rate": season_metrics["false_positive_rate"],
        "season_false_negative_rate": season_metrics["false_negative_rate"],
        "season_avg_false_positive_count": season_metrics["avg_false_positive_count"],
        "season_avg_false_negative_count": season_metrics["avg_false_negative_count"],
        "season_avg_predicted_count": season_metrics["avg_predicted_count"],
        "counts": {
            "category_rows": len(category_rows),
            "color_rows": len(color_rows),
            "occasion_rows": len(occasion_rows),
            "season_rows": len(season_rows),
        },
    }


def summarize_quality_results(results: list[dict[str, Any]], failures_limit: int, topk: int) -> dict[str, Any]:
    valid_results = [item for item in results if "error" not in item]
    error_results = [item for item in results if "error" in item]
    elapsed_values = [item["elapsed_sec"] for item in results if isinstance(item.get("elapsed_sec"), (int, float))]

    category_rows = [item for item in valid_results if has_expected_value(item["expected"]["category"])]
    color_rows = [item for item in valid_results if has_expected_value(item["expected"]["color"])]
    occasion_rows = [item for item in valid_results if has_expected_value(item["expected"]["occasions"])]
    season_rows = [item for item in valid_results if has_expected_value(item["expected"]["seasons"])]

    category_confusion = defaultdict(int)
    for item in category_rows:
        expected = item["expected"]["category"]
        predicted = item["actual"]["category"]
        if expected != predicted:
            category_confusion[(expected, predicted)] += 1

    by_group = defaultdict(list)
    by_subgroup = defaultdict(list)
    for item in valid_results:
        by_group[item["group"]].append(item)
        subgroup_key = f'{item["group"]}/{item.get("subgroup") or "_"}'
        by_subgroup[subgroup_key].append(item)

    failed_items = {
        "category": [
            {
                "file": item["file"],
                "group": item["group"],
                "subgroup": item.get("subgroup", ""),
                "expected": item["expected"]["category"],
                "predicted": item["actual"]["category"],
                f"top{topk}": item["category_topk_values"],
                "debug": item.get("debug"),
            }
            for item in category_rows
            if not item["category_pass"]
        ][:failures_limit],
        "color": [
            {
                "file": item["file"],
                "group": item["group"],
                "subgroup": item.get("subgroup", ""),
                "expected": item["expected"]["color"],
                "predicted": item["actual"]["colors"],
            }
            for item in color_rows
            if item["color_eval"]["hit_count"] == 0
        ][:failures_limit],
        "occasion": [
            {
                "file": item["file"],
                "group": item["group"],
                "subgroup": item.get("subgroup", ""),
                "expected": item["expected"]["occasions"],
                "predicted": item["actual"]["occasions"],
                "main_category_key": item["actual"]["raw"].get("mainCategoryKey"),
                "name": item["actual"]["raw"].get("name"),
                "evaluation": item["occasion_eval"],
                "debug": item.get("debug"),
            }
            for item in occasion_rows
            if not item["occasion_eval"]["exact_match"]
        ][:failures_limit],
        "season": [
            {
                "file": item["file"],
                "group": item["group"],
                "subgroup": item.get("subgroup", ""),
                "expected": item["expected"]["seasons"],
                "predicted": item["actual"]["seasons"],
                "main_category_key": item["actual"]["raw"].get("mainCategoryKey"),
                "name": item["actual"]["raw"].get("name"),
                "evaluation": item["season_eval"],
                "debug": item.get("debug"),
            }
            for item in season_rows
            if not item["season_eval"]["exact_match"]
        ][:failures_limit],
        "errors": error_results[:failures_limit],
    }

    color_metrics = summarize_multiselect_rows(color_rows, "color_eval")
    occasion_metrics = summarize_multiselect_rows(occasion_rows, "occasion_eval")
    season_metrics = summarize_multiselect_rows(season_rows, "season_eval")
    occasion_breakdown = build_multilabel_case_breakdown(occasion_rows, "occasion_eval")
    season_breakdown = build_multilabel_case_breakdown(season_rows, "season_eval")

    return {
        "total_rows": len(results),
        "valid_rows": len(valid_results),
        "error_rows": len(error_results),
        "evaluation_policy": {
            "category": {"primary_metric": "category_top1_accuracy", "secondary_metric": f"category_top{topk}_hit_rate"},
            "color": {"primary_metric": "color_hit_rate"},
            "occasion": {
                "selection_type": "multi_output",
                "pass_rule": "exact_match_only",
                "primary_metric": "occasion_pass_rate",
                "equivalent_metric": "occasion_exact_match_rate",
                "supplementary_metrics": [
                    "occasion_hit_rate",
                    "occasion_partial_hit_rate",
                    "occasion_avg_precision",
                    "occasion_avg_recall",
                    "occasion_false_positive_rate",
                    "occasion_false_negative_rate",
                ],
            },
            "season": {
                "selection_type": "multi_output",
                "pass_rule": "exact_match_only",
                "primary_metric": "season_pass_rate",
                "equivalent_metric": "season_exact_match_rate",
                "supplementary_metrics": [
                    "season_hit_rate",
                    "season_partial_hit_rate",
                    "season_avg_precision",
                    "season_avg_recall",
                    "season_false_positive_rate",
                    "season_false_negative_rate",
                ],
            },
        },
        "category_top1_accuracy": safe_ratio(sum(1 for item in category_rows if item["category_pass"]), len(category_rows)),
        "category_topk_hit_rate": safe_ratio(sum(1 for item in category_rows if item["category_topk_hit"]), len(category_rows)),
        "color_hit_rate": color_metrics["hit_rate"],
        "occasion_pass_rate": occasion_metrics["pass_rate"],
        "occasion_hit_rate": occasion_metrics["hit_rate"],
        "occasion_exact_match_rate": occasion_metrics["exact_match_rate"],
        "occasion_partial_hit_rate": occasion_metrics["partial_hit_rate"],
        "occasion_avg_precision": occasion_metrics["avg_precision"],
        "occasion_avg_recall": occasion_metrics["avg_recall"],
        "occasion_false_positive_rate": occasion_metrics["false_positive_rate"],
        "occasion_false_negative_rate": occasion_metrics["false_negative_rate"],
        "occasion_avg_false_positive_count": occasion_metrics["avg_false_positive_count"],
        "occasion_avg_false_negative_count": occasion_metrics["avg_false_negative_count"],
        "occasion_avg_predicted_count": occasion_metrics["avg_predicted_count"],
        "season_pass_rate": season_metrics["pass_rate"],
        "season_hit_rate": season_metrics["hit_rate"],
        "season_exact_match_rate": season_metrics["exact_match_rate"],
        "season_partial_hit_rate": season_metrics["partial_hit_rate"],
        "season_avg_precision": season_metrics["avg_precision"],
        "season_avg_recall": season_metrics["avg_recall"],
        "season_false_positive_rate": season_metrics["false_positive_rate"],
        "season_false_negative_rate": season_metrics["false_negative_rate"],
        "season_avg_false_positive_count": season_metrics["avg_false_positive_count"],
        "season_avg_false_negative_count": season_metrics["avg_false_negative_count"],
        "season_avg_predicted_count": season_metrics["avg_predicted_count"],
        "occasion_case_breakdown": occasion_breakdown,
        "season_case_breakdown": season_breakdown,
        "counts": {
            "category_rows": len(category_rows),
            "color_rows": len(color_rows),
            "occasion_rows": len(occasion_rows),
            "season_rows": len(season_rows),
        },
        "by_group": {key: build_quality_bucket_summary(items) for key, items in sorted(by_group.items())},
        "by_subgroup": {key: build_quality_bucket_summary(items) for key, items in sorted(by_subgroup.items())},
        "category_confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(category_confusion.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ][:failures_limit],
        "failed_items": failed_items,
        "time": {
            "total_elapsed_sec": round(sum(elapsed_values), 4),
            "avg_elapsed_sec_per_item": round(safe_ratio(sum(elapsed_values), len(elapsed_values)), 4),
            "median_elapsed_sec": round(statistics.median(elapsed_values), 4) if elapsed_values else 0.0,
            "p95_elapsed_sec": round(sorted(elapsed_values)[max(0, int(len(elapsed_values) * 0.95) - 1)], 4) if elapsed_values else 0.0,
            "max_elapsed_sec": round(max(elapsed_values), 4) if elapsed_values else 0.0,
            "min_elapsed_sec": round(min(elapsed_values), 4) if elapsed_values else 0.0,
        },
    }


def iter_validation_cases(dataset_dir: Path) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    for category in VALIDATION_EXPECTED_RULES:
        category_dir = dataset_dir / category
        if not category_dir.exists():
            continue
        for file_path in sorted(category_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                cases.append(ValidationCase(category=category, file_path=file_path))
    return cases


def evaluate_validation_result(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    expected_ok = expected["ok"]
    actual_ok = actual.get("ok")
    if actual_ok != expected_ok:
        return False, f"預期 ok={expected_ok}，實際 ok={actual_ok}"

    if not expected_ok:
        expected_reason = expected.get("reason")
        actual_reason = actual.get("reason")
        if expected_reason != actual_reason:
            return False, f"預期 reason={expected_reason}，實際 reason={actual_reason}"

    return True, "pass"


def build_short_actual_summary(actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": actual.get("ok"),
        "reason": actual.get("reason"),
        "route": actual.get("route"),
        "coarseType": actual.get("coarseType"),
        "name": actual.get("name"),
        "category": actual.get("category"),
        "color": actual.get("color"),
        "detected": actual.get("detected"),
        "bbox": actual.get("bbox"),
        "validation": actual.get("validation"),
    }


def evaluate_validation_case(case: ValidationCase, runner: PredictRunner) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        actual = runner.predict(case.file_path)
        passed, message = evaluate_validation_result(case.expected, actual)
    except Exception as exc:
        actual = {"ok": None, "error": str(exc)}
        passed = False
        message = f"predict_failed: {exc}"

    return {
        "category": case.category,
        "file": case.file_path.name,
        "file_path": str(case.file_path),
        "expected": case.expected,
        "passed": passed,
        "message": message,
        "elapsed_sec": round(time.perf_counter() - started_at, 4),
        "actual": actual,
        "actual_summary": build_short_actual_summary(actual) if isinstance(actual, dict) else {"raw": str(actual)},
    }


def summarize_validation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    category_stats: dict[str, dict[str, int]] = {
        category: {"total": 0, "pass": 0, "fail": 0}
        for category in VALIDATION_EXPECTED_RULES
    }
    elapsed_values: list[float] = []
    total_pass = 0
    total_fail = 0

    for item in results:
        category_stats[item["category"]]["total"] += 1
        if item["passed"]:
            category_stats[item["category"]]["pass"] += 1
            total_pass += 1
        else:
            category_stats[item["category"]]["fail"] += 1
            total_fail += 1
        if isinstance(item.get("elapsed_sec"), (int, float)):
            elapsed_values.append(float(item["elapsed_sec"]))

    return {
        "total_cases": len(results),
        "total_pass": total_pass,
        "total_fail": total_fail,
        "pass_rate": safe_ratio(total_pass, len(results)),
        "category_stats": category_stats,
        "time": {
            "total_elapsed_sec": round(sum(elapsed_values), 4),
            "avg_elapsed_sec_per_item": round(safe_ratio(sum(elapsed_values), len(elapsed_values)), 4),
            "max_elapsed_sec": round(max(elapsed_values), 4) if elapsed_values else 0.0,
            "min_elapsed_sec": round(min(elapsed_values), 4) if elapsed_values else 0.0,
        },
        "failed_items": [
            {
                "category": item["category"],
                "file": item["file"],
                "message": item["message"],
                "expected": item["expected"],
                "actual_summary": item["actual_summary"],
            }
            for item in results
            if not item["passed"]
        ],
    }


def run_quality(args: argparse.Namespace, runner: PredictRunner, dataset_dir: Path, report_file: Path) -> int:
    labels_file = Path(args.labels_file).resolve()
    if not labels_file.exists():
        print(f"[ERROR] 找不到 labels 檔案：{labels_file}")
        return 1

    rows = load_quality_labels(labels_file)
    rows = filter_quality_labels(rows, parse_csv_list(args.groups), parse_csv_list(args.subgroups))
    if not rows:
        print("[ERROR] 沒有可評估的 quality 標註資料。")
        return 1

    if runner.kind == "pipeline" and not args.skip_warmup:
        warmup = runner.warmup()
    else:
        warmup = None

    print("=" * 80)
    print("服飾屬性品質驗證")
    print(f"RUNNER        : {runner.kind}")
    print(f"DATASET_DIR   : {dataset_dir}")
    print(f"LABELS_FILE   : {labels_file}")
    print(f"ROWS          : {len(rows)}")
    print(f"TOPK          : {args.topk}")
    if args.groups:
        print(f"GROUPS        : {args.groups}")
    if args.subgroups:
        print(f"SUBGROUPS     : {args.subgroups}")
    if runner.kind == "api":
        print(f"API_URL       : {args.api_url}")
    print("=" * 80)

    script_started_at = time.perf_counter()
    results = [evaluate_quality_case(row, dataset_dir, runner, topk=args.topk) for row in rows]
    summary = summarize_quality_results(results, failures_limit=args.failures_limit, topk=args.topk)
    script_elapsed_sec = round(time.perf_counter() - script_started_at, 4)

    report = {
        "task": "quality",
        "runner": runner.kind,
        "dataset_dir": str(dataset_dir),
        "labels_file": str(labels_file),
        "api_url": args.api_url if runner.kind == "api" else None,
        "topk": args.topk,
        "filters": {
            "groups": sorted(parse_csv_list(args.groups) or []),
            "subgroups": sorted(parse_csv_list(args.subgroups) or []),
        },
        "warmup": warmup,
        "script_elapsed_sec": script_elapsed_sec,
        "summary": summary,
        "items": results,
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport saved to: {report_file.resolve()}")
    return 0 if summary["error_rows"] == 0 else 2


def run_validation(args: argparse.Namespace, runner: PredictRunner, dataset_dir: Path, report_file: Path) -> int:
    cases = iter_validation_cases(dataset_dir)
    if not cases:
        print(f"[ERROR] 找不到 validation 測試資料：{dataset_dir}")
        return 1

    if runner.kind == "pipeline" and not args.skip_warmup:
        warmup = runner.warmup()
    else:
        warmup = None

    print("=" * 80)
    print("服飾輸入有效性驗證")
    print(f"RUNNER        : {runner.kind}")
    print(f"DATASET_DIR   : {dataset_dir}")
    print(f"CASES         : {len(cases)}")
    if runner.kind == "api":
        print(f"API_URL       : {args.api_url}")
    print("=" * 80)

    results = [evaluate_validation_case(case, runner) for case in cases]
    summary = summarize_validation_results(results)
    report = {
        "task": "validation",
        "runner": runner.kind,
        "dataset_dir": str(dataset_dir),
        "api_url": args.api_url if runner.kind == "api" else None,
        "warmup": warmup,
        "summary": summary,
        "items": results,
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport saved to: {report_file.resolve()}")
    return 0 if summary["total_fail"] == 0 else 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.task, args.dataset_dir)
    if not dataset_dir.exists():
        print(f"[ERROR] 找不到測試集資料夾：{dataset_dir}")
        return 1

    report_file = resolve_report_file_path(args.report_file, dataset_dir)
    runner = build_runner(args)

    if args.task == "validation":
        return run_validation(args, runner, dataset_dir, report_file)
    return run_quality(args, runner, dataset_dir, report_file)


if __name__ == "__main__":
    raise SystemExit(main())