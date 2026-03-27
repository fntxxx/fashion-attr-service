import csv
from pathlib import Path

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")
LABELS_FILE = Path("artifacts/labels/labels_category_generated.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_dataset_files():
    files = []
    for p in DATASET_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def main():
    dataset_files = collect_dataset_files()
    dataset_name_map = {p.name.lower(): p for p in dataset_files}
    dataset_stem_map = {}

    for p in dataset_files:
        dataset_stem_map.setdefault(p.stem.lower(), []).append(p)

    missing = []
    with open(LABELS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_file = row["file"].strip()
            lower_name = label_file.lower()
            stem = Path(label_file).stem.lower()

            exact = dataset_name_map.get(lower_name)
            stem_matches = dataset_stem_map.get(stem, [])

            if exact is None:
                missing.append({
                    "label_file": label_file,
                    "stem_matches": [str(p) for p in stem_matches]
                })

    print(f"missing count = {len(missing)}")
    print()

    for item in missing:
        print("LABEL:", item["label_file"])
        if item["stem_matches"]:
            print("  stem matches:")
            for p in item["stem_matches"]:
                print("   -", p)
        else:
            print("  stem matches: NONE")
        print()


if __name__ == "__main__":
    main()