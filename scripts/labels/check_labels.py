from pathlib import Path
import csv

with open("artifacts/labels/labels.csv", "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("rows =", len(rows))
print("first row =", rows[0])
print("last row =", rows[-1])