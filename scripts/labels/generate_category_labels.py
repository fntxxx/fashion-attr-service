import csv
from pathlib import Path

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset\category")
OUTPUT_FILE = Path("artifacts/labels/labels_category_generated.csv")

CATEGORY_DEFAULTS = {
    "top": {
        "expected_occasions": "campus_casual",
        "expected_seasons": "spring|autumn",
    },
    "outer": {
        "expected_occasions": "business_casual|campus_casual",
        "expected_seasons": "autumn|winter",
    },
    "pants": {
        "expected_occasions": "campus_casual",
        "expected_seasons": "spring|autumn",
    },
    "skirt": {
        "expected_occasions": "campus_casual",
        "expected_seasons": "spring|autumn",
    },
    "dress": {
        "expected_occasions": "social",
        "expected_seasons": "spring|summer",
    },
    "shoes": {
        "expected_occasions": "social|campus_casual",
        "expected_seasons": "spring|summer|autumn",
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    rows = []

    for category_dir in sorted(DATASET_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name.strip()
        defaults = CATEGORY_DEFAULTS.get(category)
        if not defaults:
            print(f"skip unknown category folder: {category}")
            continue

        for image_path in sorted(category_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rows.append({
                "file": image_path.name,
                "group": "category",
                "expected_category": category,
                "expected_color": "",
                "expected_occasions": defaults["expected_occasions"],
                "expected_seasons": defaults["expected_seasons"],
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "group",
                "expected_category",
                "expected_color",
                "expected_occasions",
                "expected_seasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"generated rows = {len(rows)}")
    print(f"saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()