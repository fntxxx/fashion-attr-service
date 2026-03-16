from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from services.extract_color import extract_color

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
LABELS_FILE = Path("labels_full_template.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TARGET_SUBGROUPS = {
    "pattern",
    "elegant_purple",
    "natural_green",
    "warm_orange_red",
}


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
    base = Path(filename).stem

    for path in DATASET_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem == base:
            return path

    return None


def subgroup_to_expected_color(subgroup: str) -> Optional[str]:
    mapping = {
        "earth_brown": "咖啡色系",
        "rose_pink": "紅色系",
        "pattern": "花紋圖案",
        "light_beige": "米色系",
        "warm_orange_red": "紅色系",
        "neutral_gray": "灰色系",
        "elegant_purple": "紫色系",
        "natural_green": "綠色系",
        "fresh_blue": "藍色系",
        "dark_gray_black": "黑色系",
    }
    return mapping.get(subgroup)


def main():
    labels = load_labels()

    target_rows = [
        row
        for row in labels
        if row["group"] == "color" and row["subgroup"] in TARGET_SUBGROUPS
    ]

    total = 0
    hit = 0
    missing = 0

    subgroup_total = defaultdict(int)
    subgroup_hit = defaultdict(int)

    print("=== Color Subset Test ===")

    for row in target_rows:
        image_path = find_image_file(row["file"])
        if image_path is None:
            print(f'[MISS] subgroup={row["subgroup"]:<16} file={row["file"]:<36} 找不到圖片')
            missing += 1
            continue

        expected = subgroup_to_expected_color(row["subgroup"])
        if not expected:
            print(f'[SKIP] subgroup={row["subgroup"]:<16} file={row["file"]:<36} 沒有對應 expected color')
            continue

        subgroup_total[row["subgroup"]] += 1

        with Image.open(image_path) as img:
            pred = extract_color(img)

        ok = pred == expected

        print(
            f'[{"OK" if ok else "NG"}] '
            f'subgroup={row["subgroup"]:<16} '
            f'file={row["file"]:<36} '
            f'pred={pred:<8} '
            f'expected={expected}'
        )

        total += 1
        if ok:
            hit += 1
            subgroup_hit[row["subgroup"]] += 1

    print()
    print("=== Subgroup Summary ===")
    for subgroup in sorted(TARGET_SUBGROUPS):
        sg_total = subgroup_total[subgroup]
        sg_hit = subgroup_hit[subgroup]
        sg_acc = (sg_hit / sg_total) if sg_total else 0.0
        print(f"{subgroup:<16} total={sg_total:<3d} hit={sg_hit:<3d} acc={sg_acc:.4f}")

    print()
    print(f"tested={total}, hit={hit}, miss={missing}")
    print(f"accuracy={hit / total:.4f}" if total else "accuracy=0.0000")


if __name__ == "__main__":
    main()