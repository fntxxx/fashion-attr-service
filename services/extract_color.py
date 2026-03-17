from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np
from PIL import Image


def _resize_for_color(image: Image.Image, max_side: int = 256) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


def _rgb_to_hsv_np(arr: np.ndarray) -> np.ndarray:
    """
    arr: uint8 RGB, shape (N, 3)
    return: float HSV
      H: 0~360
      S: 0~1
      V: 0~1
    """
    rgb = arr.astype(np.float32) / 255.0
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)

    mask = delta > 1e-6

    r_mask = (cmax == r) & mask
    g_mask = (cmax == g) & mask
    b_mask = (cmax == b) & mask

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4

    h = h * 60.0

    s = np.zeros_like(cmax)
    nonzero = cmax > 1e-6
    s[nonzero] = delta[nonzero] / cmax[nonzero]

    v = cmax

    return np.stack([h, s, v], axis=1)


def _center_weight_mask(h: int, w: int) -> np.ndarray:
    """
    中央權重較高，降低背景干擾。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dy = (yy - cy) / max(h / 2.0, 1.0)
    dx = (xx - cx) / max(w / 2.0, 1.0)
    dist = np.sqrt(dx * dx + dy * dy)
    weight = 1.15 - np.clip(dist, 0.0, 1.0)
    return np.clip(weight, 0.15, 1.15)


def _estimate_foreground(image: Image.Image) -> np.ndarray:
    """
    便宜版前景估計：
    - 用邊界顏色估背景
    - 排除太接近背景的像素
    """
    img = _resize_for_color(image, 256)
    arr = np.array(img).astype(np.int16)
    h, w, _ = arr.shape

    border = np.concatenate(
        [
            arr[:8, :, :].reshape(-1, 3),
            arr[-8:, :, :].reshape(-1, 3),
            arr[:, :8, :].reshape(-1, 3),
            arr[:, -8:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = border.mean(axis=0)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))

    # 與背景差太小的點視為背景
    fg_mask = diff > 18

    # 如果前景太少，退回全圖
    if fg_mask.mean() < 0.08:
        fg_mask = np.ones((h, w), dtype=bool)

    return fg_mask


def _weighted_hsv_hist(hsv: np.ndarray, weights: np.ndarray) -> Counter:
    """
    把 HSV 粗量化後做加權統計，提升穩定性。
    """
    h_bin = np.floor(hsv[:, 0] / 15).astype(int)     # 24 bins
    s_bin = np.floor(hsv[:, 1] / 0.12).astype(int)   # 約 9 bins
    v_bin = np.floor(hsv[:, 2] / 0.12).astype(int)   # 約 9 bins

    counter: Counter = Counter()
    for hb, sb, vb, wt in zip(h_bin, s_bin, v_bin, weights):
        counter[(hb, sb, vb)] += float(wt)

    return counter


def _is_pattern(hsv: np.ndarray, weights: np.ndarray) -> bool:
    total = float(np.sum(weights))
    if total <= 0:
        return False

    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]

    colorful = weights[sat >= 0.20]
    colorful_ratio = float(np.sum(colorful) / total) if total else 0.0

    bright = weights[val >= 0.35]
    bright_ratio = float(np.sum(bright) / total) if total else 0.0

    hue_mask = sat >= 0.18
    if np.sum(hue_mask) < 20:
        return False

    hue_bins = np.floor(hue[hue_mask] / 20).astype(int)
    hue_counter: Counter = Counter()
    for hb, wt in zip(hue_bins, weights[hue_mask]):
        hue_counter[int(hb)] += float(wt)

    sorted_bins = hue_counter.most_common()
    dominant_bin = sorted_bins[0][0] if sorted_bins else 0
    dominant = sorted_bins[0][1] if sorted_bins else 0.0
    second = sorted_bins[1][1] if len(sorted_bins) > 1 else 0.0
    third = sorted_bins[2][1] if len(sorted_bins) > 2 else 0.0

    dominant_ratio = dominant / total if total else 1.0
    top2_ratio = (dominant + second) / total if total else 1.0
    top3_ratio = (dominant + second + third) / total if total else 1.0
    active_bins = len(sorted_bins)

    # 相鄰色相群聚保護（含環狀相鄰）
    # 例如：
    # - 綠色群：1,2,3
    # - 紅橘群：17,0,1
    neighbor_bins = {
        (dominant_bin - 1) % 18,
        dominant_bin,
        (dominant_bin + 1) % 18,
    }
    neighbor_ratio = (
        sum(hue_counter.get(b, 0.0) for b in neighbor_bins) / total
        if total else 1.0
    )

    # 只對暖色主群做相鄰 hue 保護，避免紅橘鞋款被誤抓成 pattern。
    # bins:
    # 17 -> 340~360
    # 0  ->   0~20
    # 1  ->  20~40
    is_warm_hue_cluster = dominant_bin in {17, 0, 1}

    if (
        is_warm_hue_cluster
        and neighbor_ratio >= 0.55
        and dominant_ratio >= 0.30
        and colorful_ratio >= 0.35
    ):
        return False

    # 單色保護 1
    if active_bins <= 3 and top2_ratio >= 0.72 and colorful_ratio >= 0.35:
        return False

    # 單色保護 2
    if active_bins <= 4 and dominant_ratio >= 0.22 and top3_ratio >= 0.82:
        return False

    # 原本主規則
    if colorful_ratio >= 0.30 and bright_ratio >= 0.30 and dominant_ratio <= 0.58:
        return True

    # 偏暗花紋：顏色夠分散，但整體亮度偏低
    if colorful_ratio >= 0.50 and bright_ratio >= 0.18 and dominant_ratio <= 0.45:
        return True

    # 低彩度亮色花紋：像淺色條紋、淡色印花
    if colorful_ratio >= 0.20 and bright_ratio >= 0.85 and dominant_ratio <= 0.12:
        return True

    return False


def _classify_color_from_hsv(hsv: np.ndarray, weights: np.ndarray) -> str:
    total = float(np.sum(weights))
    if total <= 0:
        return "灰色系"

    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]

    # 深色先判
    very_dark_ratio = float(np.sum(weights[val <= 0.22]) / total)
    dark_ratio = float(np.sum(weights[val <= 0.30]) / total)
    low_sat_ratio = float(np.sum(weights[sat <= 0.16]) / total)

    # ===== 淡粉（低飽和但偏紅）優先攔截 =====
    avg_v = float(np.average(val, weights=weights))
    avg_s = float(np.average(sat, weights=weights))
    redish_ratio = float(
        np.sum(weights[((hue >= 320) | (hue < 20))]) / total
    )

    pinkish_ratio = float(
        np.sum(weights[
            ((hue >= 320) | (hue < 20)) &
            (sat >= 0.08) & (sat <= 0.70) &
            (val >= 0.60)
        ]) / total
    )

    if pinkish_ratio >= 0.55:
        return "粉紅色系"

    # 白 / 米 / 灰
    if low_sat_ratio >= 0.60:
        avg_v = float(np.average(val, weights=weights))
        avg_s = float(np.average(sat, weights=weights))
        beigeish_ratio = float(
            np.sum(weights[(hue >= 30) & (hue <= 70)]) / total
        )

        if avg_v <= 0.26 and dark_ratio >= 0.80:
            return "黑色系"

        if avg_v >= 0.88 and avg_s <= 0.08:
            return "白色系"

        # 米色通常亮、低到中低飽和，且偏黃棕區
        if avg_v >= 0.58 and avg_s <= 0.22 and beigeish_ratio >= 0.35:
            return "米色系"

        return "灰色系"

    # 有彩色時，用主色相判斷
    hist = _weighted_hsv_hist(hsv, weights)
    top_bin, _ = hist.most_common(1)[0]
    hb, sb, vb = top_bin

    h_center = hb * 15 + 7.5
    s_center = sb * 0.12 + 0.06
    v_center = vb * 0.12 + 0.06

    # 卡其 / 咖啡
    if 15 <= h_center <= 60 and s_center <= 0.45:

        if v_center >= 0.60 and s_center <= 0.30:
            return "米色系"

        if v_center >= 0.45:
            return "卡其色系"

        return "咖啡色系"

    # 紅 / 橘 / 粉
    if (h_center >= 345 or h_center < 20):
        # rose pink：高亮 + 中低飽和
        if 0.10 <= s_center <= 0.50 and v_center >= 0.60:
            return "粉紅色系"

        # 深暖棕常被量化到 0~20 度，先擋掉
        if 0.28 <= s_center <= 0.55 and v_center <= 0.52:
            return "咖啡色系"

        return "紅色系"

    if 20 <= h_center < 45:

        # 暗棕，即使飽和稍高，也先視為咖啡
        if v_center <= 0.28:
            return "咖啡色系"

        if s_center <= 0.56 and v_center <= 0.62:
            return "咖啡色系"

        return "紅色系"

    if 45 <= h_center < 85:
        # 黃綠偏卡其的情況
        if s_center <= 0.32 and v_center >= 0.5:
            return "卡其色系"
        return "綠色系"

    if 85 <= h_center < 170:
        return "綠色系"

    # blue / denim
    if 170 <= h_center < 255:

        # washed denim
        if v_center >= 0.70 and s_center <= 0.30:
            return "藍色系"

        return "藍色系"

    if 255 <= h_center < 320:
        return "紫色系"

    if 320 <= h_center < 345:
        if 0.10 <= s_center <= 0.46 and v_center >= 0.58:
            return "粉紅色系"
        return "紅色系"

    return "灰色系"


def extract_color(image: Image.Image) -> str:
    """
    回傳色系：
    白色系 / 米色系 / 黑色系 / 灰色系 / 卡其色系 / 咖啡色系 /
    紅色系 / 粉紅色系 / 綠色系 / 藍色系 / 紫色系 / 花紋圖案
    """
    img = _resize_for_color(image, 256)
    arr = np.array(img)
    h, w, _ = arr.shape

    fg_mask = _estimate_foreground(img)
    center_w = _center_weight_mask(h, w)

    pixels = arr.reshape(-1, 3)
    fg = fg_mask.reshape(-1)
    weights = center_w.reshape(-1)

    pixels = pixels[fg]
    weights = weights[fg]

    if len(pixels) == 0:
        return "灰色系"

    hsv = _rgb_to_hsv_np(pixels)

    if _is_pattern(hsv, weights):
        return "花紋圖案"

    return _classify_color_from_hsv(hsv, weights)