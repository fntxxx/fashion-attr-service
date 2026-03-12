from __future__ import annotations

import json
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


API_URL = "http://127.0.0.1:7860/predict"
DEFAULT_DATASET_DIR = Path(r"D:\DevData\ai_testset")
REPORT_FILE = "test_attr_api_report.json"
TIMEOUT_SECONDS = 60

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

EXPECTED_RULES = {
    "valid_product": {"ok": True},
    "flatlay": {"ok": True},
    "person_outfit": {"ok": False, "reason": "not_fashion_image"},
    "multi_item": {"ok": False, "reason": "not_fashion_image"},
    "invalid": {"ok": False, "reason": "not_fashion_image"},
}


@dataclass
class TestCase:
    category: str
    file_path: Path

    @property
    def expected(self) -> dict[str, Any]:
        return EXPECTED_RULES[self.category]


def iter_test_cases(dataset_dir: Path) -> list[TestCase]:
    cases: list[TestCase] = []

    for category in EXPECTED_RULES.keys():
        category_dir = dataset_dir / category
        if not category_dir.exists():
            print(f"[WARN] 找不到資料夾：{category_dir}")
            continue

        for file_path in sorted(category_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            cases.append(TestCase(category=category, file_path=file_path))

    return cases


def call_predict_api(file_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "application/octet-stream"

    with file_path.open("rb") as f:
        files = {
            "image": (file_path.name, f, mime_type),
        }
        response = requests.post(API_URL, files=files, timeout=TIMEOUT_SECONDS)

    response.raise_for_status()
    return response.json()


def evaluate_result(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    expected_ok = expected["ok"]
    actual_ok = actual.get("ok")

    if actual_ok != expected_ok:
        return (
            False,
            f"預期 ok={expected_ok}，實際 ok={actual_ok}",
        )

    if not expected_ok:
        expected_reason = expected.get("reason")
        actual_reason = actual.get("reason")
        if expected_reason != actual_reason:
            return (
                False,
                f"預期 reason={expected_reason}，實際 reason={actual_reason}",
            )

    return True, "pass"


def build_short_actual_summary(actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": actual.get("ok"),
        "reason": actual.get("reason"),
        "route": actual.get("route"),
        "coarseType": actual.get("coarseType"),
        "mainCategoryKey": actual.get("mainCategoryKey"),
        "categoryKey": actual.get("categoryKey"),
        "colorTone": actual.get("colorTone"),
        "detected": actual.get("detected"),
        "bbox": actual.get("bbox"),
        "validation": actual.get("validation"),
    }


def main() -> int:
    dataset_dir = DEFAULT_DATASET_DIR
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])

    dataset_dir = dataset_dir.resolve()

    if not dataset_dir.exists():
        print(f"[ERROR] 找不到測試集資料夾：{dataset_dir}")
        return 1

    print("=" * 80)
    print("衣物屬性辨識 API 測試")
    print(f"API_URL      : {API_URL}")
    print(f"DATASET_DIR  : {dataset_dir}")
    print("=" * 80)

    cases = iter_test_cases(dataset_dir)
    if not cases:
        print("[ERROR] 沒有找到任何可測試的圖片。")
        return 1

    category_stats: dict[str, dict[str, int]] = {
        category: {"total": 0, "pass": 0, "fail": 0}
        for category in EXPECTED_RULES.keys()
    }

    report_items: list[dict[str, Any]] = []
    total_pass = 0
    total_fail = 0

    for index, case in enumerate(cases, start=1):
        category_stats[case.category]["total"] += 1

        print(f"\n[{index}/{len(cases)}] 測試中：{case.category}/{case.file_path.name}")

        try:
            actual = call_predict_api(case.file_path)
            passed, message = evaluate_result(case.expected, actual)
        except requests.RequestException as e:
            passed = False
            message = f"HTTP 錯誤：{e}"
            actual = {"ok": None, "error": str(e)}
        except Exception as e:
            passed = False
            message = f"例外錯誤：{e}"
            actual = {"ok": None, "error": str(e)}

        if passed:
            total_pass += 1
            category_stats[case.category]["pass"] += 1
            status = "PASS"
        else:
            total_fail += 1
            category_stats[case.category]["fail"] += 1
            status = "FAIL"

        short_actual = build_short_actual_summary(actual) if isinstance(actual, dict) else {"raw": str(actual)}

        print(f"  [{status}] {message}")
        print(f"  expected: {case.expected}")
        print(f"  actual  : {json.dumps(short_actual, ensure_ascii=False)}")

        report_items.append(
            {
                "category": case.category,
                "file": case.file_path.name,
                "file_path": str(case.file_path),
                "expected": case.expected,
                "passed": passed,
                "message": message,
                "actual": actual,
            }
        )

    print("\n" + "=" * 80)
    print("分類統計")
    print("=" * 80)

    for category, stats in category_stats.items():
        print(
            f"{category:15} "
            f"{stats['pass']}/{stats['total']} pass"
            f" | fail={stats['fail']}"
        )

    print("-" * 80)
    print(f"TOTAL: {total_pass}/{len(cases)} pass | fail={total_fail}")

    report = {
        "api_url": API_URL,
        "dataset_dir": str(dataset_dir),
        "total_cases": len(cases),
        "total_pass": total_pass,
        "total_fail": total_fail,
        "category_stats": category_stats,
        "items": report_items,
    }

    report_path = Path(REPORT_FILE).resolve()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n測試報告已輸出：{report_path}")

    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())