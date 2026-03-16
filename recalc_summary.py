import json

with open("test_attr_quality_report.json", "r", encoding="utf-8") as f:
    report = json.load(f)

items = report["items"]

category_pass = 0
valid = 0

confusion = {}

for r in items:
    if "error" in r:
        continue

    valid += 1

    expected = r["expected"]["category"]
    actual = r["actual"]["category"]

    if expected == actual:
        category_pass += 1
    else:
        key = (expected, actual)
        confusion[key] = confusion.get(key, 0) + 1

print("valid =", valid)
print("accuracy =", category_pass / valid)

print("\nconfusion:")
for k, v in confusion.items():
    print(k, v)