import csv

with open("labels_category_generated.csv", "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("rows =", len(rows))
print("first row =", rows[0])
print("last row =", rows[-1])