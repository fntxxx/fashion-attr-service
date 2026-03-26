from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from services.extract_color import (
    _resize_for_color,
    _estimate_foreground,
    _center_weight_mask,
    _rgb_to_hsv_np,
    _weighted_hsv_hist,
)

DATASET_DIR = Path(r"D:\DevData\attr_quality_testset")

TARGET_FILES = [
    "color_rose_pink_dress_03.jpg",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_image_file(filename: str) -> Path | None:
    base = Path(filename).stem

    for path in DATASET_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem == base:
            return path

    return None


def main():
    for fname in TARGET_FILES:
        path = find_image_file(fname)
        if path is None:
            print(f"[MISS] {fname}")
            continue

        img = Image.open(path).convert("RGB")
        img = _resize_for_color(img, 256)

        arr = np.array(img)
        h, w, _ = arr.shape

        fg_mask = _estimate_foreground(img)
        center_w = _center_weight_mask(h, w)

        pixels = arr.reshape(-1, 3)
        fg = fg_mask.reshape(-1)
        weights = center_w.reshape(-1)

        pixels = pixels[fg]
        weights = weights[fg]

        hsv = _rgb_to_hsv_np(pixels)

        total = float(np.sum(weights))
        hue = hsv[:, 0]
        sat = hsv[:, 1]
        val = hsv[:, 2]

        colorful_ratio = float(np.sum(weights[sat >= 0.20]) / total) if total else 0.0
        bright_ratio = float(np.sum(weights[val >= 0.35]) / total) if total else 0.0
        hue_mask = sat >= 0.18
        hue_mask_count = int(np.sum(hue_mask))

        if hue_mask_count > 0:
            hue_bins = np.floor(hue[hue_mask] / 20).astype(int)
            hist = {}
            for hb, wt in zip(hue_bins, weights[hue_mask]):
                hist[int(hb)] = hist.get(int(hb), 0.0) + float(wt)

            sorted_bins = sorted(hist.items(), key=lambda x: x[1], reverse=True)

            dominant = sorted_bins[0][1] if sorted_bins else 0.0
            second = sorted_bins[1][1] if len(sorted_bins) > 1 else 0.0
            third = sorted_bins[2][1] if len(sorted_bins) > 2 else 0.0

            dominant_ratio = dominant / total if total else 1.0
            top2_ratio = (dominant + second) / total if total else 1.0
            top3_ratio = (dominant + second + third) / total if total else 1.0
            active_bins = len(sorted_bins)
        else:
            sorted_bins = []
            dominant_ratio = 1.0
            top2_ratio = 1.0
            top3_ratio = 1.0
            active_bins = 0

        low_sat_ratio = float(np.sum(weights[sat <= 0.16]) / total) if total else 0.0
        very_dark_ratio = float(np.sum(weights[val <= 0.22]) / total) if total else 0.0
        dark_ratio = float(np.sum(weights[val <= 0.30]) / total) if total else 0.0

        avg_h = float(np.average(hue, weights=weights))
        avg_s = float(np.average(sat, weights=weights))
        avg_v = float(np.average(val, weights=weights))

        hist = _weighted_hsv_hist(hsv, weights)
        (hb, sb, vb), _ = hist.most_common(1)[0]
        h_center = hb * 15 + 7.5
        s_center = sb * 0.12 + 0.06
        v_center = vb * 0.12 + 0.06

        print("=" * 80)
        print(fname)
        print(f"colorful_ratio={colorful_ratio:.4f}")
        print(f"bright_ratio={bright_ratio:.4f}")
        print(f"dominant_ratio={dominant_ratio:.4f}")
        print(f"top2_ratio={top2_ratio:.4f}")
        print(f"top3_ratio={top3_ratio:.4f}")
        print(f"active_bins={active_bins}")
        print(f"hue_mask_count={hue_mask_count}")
        print(f"low_sat_ratio={low_sat_ratio:.4f}")
        print(f"very_dark_ratio={very_dark_ratio:.4f}")
        print(f"dark_ratio={dark_ratio:.4f}")
        print(f"avg_h={avg_h:.2f}, avg_s={avg_s:.4f}, avg_v={avg_v:.4f}")
        print(f"h_center={h_center:.2f}, s_center={s_center:.4f}, v_center={v_center:.4f}")
        print(f"top hue bins={sorted_bins[:5]}")


if __name__ == "__main__":
    main()