from __future__ import annotations

import csv
from pathlib import Path

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
OUTPUT_FILE = Path("labels_full_template.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_category_from_filename(name: str) -> str:
    lower = name.lower()

    if lower.startswith("dress_"):
        return "dress"
    if lower.startswith("outer_"):
        return "outer"
    if lower.startswith("pants_"):
        return "pants"
    if lower.startswith("skirt_"):
        return "skirt"
    if lower.startswith("shoes_"):
        return "shoes"
    if lower.startswith("top_"):
        return "top"

    return ""


def infer_color_from_filename(name: str) -> str:
    lower = name.lower()

    color_keys = [
        "light_beige",
        "dark_gray_black",
        "neutral_gray",
        "earth_brown",
        "warm_orange_red",
        "rose_pink",
        "natural_green",
        "fresh_blue",
        "elegant_purple",
        "pattern",
    ]

    for key in color_keys:
        if f"color_{key}_" in lower:
            return key

    return ""


def infer_occasions_from_filename(name: str) -> str:
    lower = name.lower()

    keys = [
        "social",
        "campus_casual",
        "business_casual",
        "professional",
    ]

    for key in keys:
        if f"occasion_{key}_" in lower:
            return key

    return ""


def infer_seasons_from_filename(name: str) -> str:
    lower = name.lower()

    keys = ["spring", "summer", "autumn", "winter"]

    for key in keys:
        if f"season_{key}_" in lower:
            return key

    return ""


def infer_defaults(group: str, filename: str) -> dict[str, str]:
    row = {
        "expected_category": "",
        "expected_color": "",
        "expected_occasions": "",
        "expected_seasons": "",
    }

    if group == "category":
        row["expected_category"] = infer_category_from_filename(filename)
    elif group == "color":
        row["expected_color"] = infer_color_from_filename(filename)
    elif group == "occasion":
        row["expected_occasions"] = infer_occasions_from_filename(filename)
    elif group == "season":
        row["expected_seasons"] = infer_seasons_from_filename(filename)

    return row


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for group_dir in sorted(DATASET_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not group_dir.is_dir():
            continue

        group = group_dir.name

        files = sorted(
            [
                p for p in group_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: str(p.relative_to(group_dir)).lower(),
        )

        for path in files:
            relative_parts = path.relative_to(group_dir).parts
            subgroup = relative_parts[0] if len(relative_parts) > 1 else ""

            row = {
                "file": path.name,
                "group": group,
                "subgroup": subgroup,
                "expected_category": "",
                "expected_color": "",
                "expected_occasions": "",
                "expected_seasons": "",
            }

            row.update(infer_defaults(group, path.name))
            rows.append(row)

    return rows


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"dataset dir not found: {DATASET_DIR}")

    rows = build_rows()

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "group",
                "subgroup",
                "expected_category",
                "expected_color",
                "expected_occasions",
                "expected_seasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"generated: {OUTPUT_FILE.resolve()}")
    print(f"total rows: {len(rows)}")


if __name__ == "__main__":
    main()