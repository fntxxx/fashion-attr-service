from __future__ import annotations

import csv
import json
import mimetypes
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# =========================
# 設定區
# =========================

API_URL = "http://127.0.0.1:7860/predict"

# 請確認這個路徑和你本機一致
DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")

# 若 labels_full_template.csv 不在專案根目錄，請改這裡
LABELS_FILE = Path("labels_full_template.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# 先測你目前最在意的 subgroup
TARGET_SUBGROUPS = {
    "earth_brown",
    "butter_yellow",
    "rose_pink",
    # 想加防退化保護時再打開
    # "light_beige",
    # "warm_orange_red",
    # "neutral_gray",
}

REQUEST_TIMEOUT = 120


# =========================
# 工具函式
# =========================

def load_labels() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(LABELS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "file": row["file"].strip(),
                    "group": row["group"].strip(),
                    "subgroup": row.get("subgroup", "").strip(),
                    "expected_color": row.get("expected_color", "").strip(),
                }
            )

    return rows


def find_image_file(filename: str) -> Optional[Path]:
    """
    根據 CSV 裡的 file 名稱，到測試集資料夾遞迴找對應圖片。
    """
    base = Path(filename).stem

    for path in DATASET_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem == base:
            return path

    return None


def parse_predicted_color(data: Dict[str, Any]) -> Optional[str]:
    """
    盡量兼容不同 API 回傳格式。
    依優先順序抓 color 結果。
    """

    # 1. 直接 color
    value = data.get("color")
    if isinstance(value, str) and value.strip():
        return value.strip()

    # 2. candidates.color[0].value
    candidates = data.get("candidates")
    if isinstance(candidates, dict):
        color_candidates = candidates.get("color")
        if isinstance(color_candidates, list) and color_candidates:
            first = color_candidates[0]
            if isinstance(first, dict):
                value = first.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()

    # 3. scores / nested 結構不處理，避免亂猜
    return None


def call_predict_api(image_path: Path) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"

    with image_path.open("rb") as f:
        files = {
            "image": (image_path.name, f, mime_type)
        }
        resp = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)

    resp.raise_for_status()
    return resp.json()


# =========================
# 主流程
# =========================

def main():
    started_at = time.time()

    labels = load_labels()

    target_rows = [
        row
        for row in labels
        if row["group"] == "color" and row["subgroup"] in TARGET_SUBGROUPS
    ]

    if not target_rows:
        print("找不到符合條件的 color subset，請檢查 TARGET_SUBGROUPS 或 labels_full_template.csv")
        return

    total = 0
    hit = 0
    missing = 0
    error = 0

    subgroup_total = defaultdict(int)
    subgroup_hit = defaultdict(int)

    print("=== Color Subset API Test ===")
    print(f"API_URL      : {API_URL}")
    print(f"DATASET_DIR  : {DATASET_DIR}")
    print(f"LABELS_FILE  : {LABELS_FILE}")
    print(f"SUBGROUPS    : {sorted(TARGET_SUBGROUPS)}")
    print()

    for row in target_rows:
        file_name = row["file"]
        subgroup = row["subgroup"]
        expected = row["expected_color"]

        image_path = find_image_file(file_name)
        if image_path is None:
            print(f'[MISS ] subgroup={subgroup:<16} file={file_name:<36} 找不到圖片')
            missing += 1
            continue

        subgroup_total[subgroup] += 1

        try:
            t0 = time.time()
            data = call_predict_api(image_path)
            elapsed = time.time() - t0

            pred = parse_predicted_color(data)
            ok = pred == expected

            print(
                f'[{"OK   " if ok else "NG   "}] '
                f'subgroup={subgroup:<16} '
                f'file={file_name:<36} '
                f'pred={str(pred):<8} '
                f'expected={expected:<8} '
                f't={elapsed:.2f}s'
            )

            total += 1
            if ok:
                hit += 1
                subgroup_hit[subgroup] += 1

        except Exception as e:
            print(
                f'[ERROR] subgroup={subgroup:<16} '
                f'file={file_name:<36} '
                f'error={type(e).__name__}: {e}'
            )
            error += 1

    elapsed_all = time.time() - started_at

    print()
    print("=== Subgroup Summary ===")
    for subgroup in sorted(TARGET_SUBGROUPS):
        sg_total = subgroup_total[subgroup]
        sg_hit = subgroup_hit[subgroup]
        sg_acc = (sg_hit / sg_total) if sg_total else 0.0
        print(
            f"{subgroup:<16} total={sg_total:<3d} hit={sg_hit:<3d} acc={sg_acc:.4f}"
        )

    print()
    print("=== Overall Summary ===")
    print(f"tested   = {total}")
    print(f"hit      = {hit}")
    print(f"missing  = {missing}")
    print(f"error    = {error}")
    print(f"accuracy = {hit / total:.4f}" if total else "accuracy = 0.0000")
    print(f"elapsed  = {elapsed_all:.2f}s")


if __name__ == "__main__":
    main()