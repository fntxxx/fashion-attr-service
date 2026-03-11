import numpy as np
from PIL import Image

from utils.color_map import map_color


def _center_weight_mask(h, w):
    y = np.linspace(-1.0, 1.0, h)
    x = np.linspace(-1.0, 1.0, w)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx * xx + yy * yy)

    weights = 1.15 - np.clip(dist, 0, 1.15)
    weights = np.clip(weights, 0.05, None)
    return weights


def _rgb_to_hsv_np(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    nonzero = maxc > 0
    s[nonzero] = diff[nonzero] / maxc[nonzero]

    mask = diff > 1e-6

    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6

    idx = (maxc == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2

    idx = (maxc == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4

    h = h * 60.0
    h = np.where(h < 0, h + 360.0, h)

    return np.stack([h, s * 255.0, v * 255.0], axis=1)


def _weighted_kmeans(rgb_pixels, weights, k=4, max_iter=12):
    n = len(rgb_pixels)
    if n == 0:
        return None, None

    if n < k:
        mean_rgb = np.average(rgb_pixels, axis=0, weights=weights)
        centers = np.array([mean_rgb], dtype=np.float32)
        labels = np.zeros(n, dtype=np.int32)
        return centers, labels

    rng = np.random.default_rng(42)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = rgb_pixels[init_idx].copy()

    for _ in range(max_iter):
        dists = np.sum((rgb_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = []
        for i in range(k):
            cluster_mask = labels == i
            if not np.any(cluster_mask):
                new_centers.append(centers[i])
                continue

            cluster_pixels = rgb_pixels[cluster_mask]
            cluster_weights = weights[cluster_mask]
            center = np.average(cluster_pixels, axis=0, weights=cluster_weights)
            new_centers.append(center)

        new_centers = np.array(new_centers, dtype=np.float32)

        if np.allclose(centers, new_centers, atol=1.0):
            centers = new_centers
            break

        centers = new_centers

    return centers, labels


def _choose_dominant_cluster(centers, labels, weights):
    if centers is None or labels is None or len(centers) == 0:
        return None, []

    hsv_centers = _rgb_to_hsv_np(centers)
    cluster_infos = []

    total_weight = float(np.sum(weights)) if len(weights) > 0 else 1.0

    for i in range(len(centers)):
        cluster_mask = labels == i
        if not np.any(cluster_mask):
            continue

        cluster_weight = float(np.sum(weights[cluster_mask]))
        ratio = cluster_weight / max(total_weight, 1e-6)

        h, s, v = hsv_centers[i]
        score = cluster_weight

        # 太白、太亮、太低彩度，通常偏背景或白條紋
        if v > 210 and s < 35:
            score *= 0.35

        # 過灰通常不是主色
        if s < 28:
            score *= 0.65

        # 中等亮度且有彩度，通常比較像布料主色
        if 45 < v < 200 and s > 40:
            score *= 1.18

        cluster_infos.append({
            "index": i,
            "rgb": centers[i],
            "hsv": hsv_centers[i],
            "weight": cluster_weight,
            "ratio": ratio,
            "dominant_score": float(score),
        })

    if not cluster_infos:
        return None, []

    cluster_infos = sorted(cluster_infos, key=lambda x: x["dominant_score"], reverse=True)
    dominant = cluster_infos[0]
    return dominant, cluster_infos


def _hue_distance(h1, h2):
    diff = abs(h1 - h2)
    return min(diff, 360.0 - diff)


def _rgb_distance(c1, c2):
    return float(np.linalg.norm(np.array(c1) - np.array(c2)))


def _is_pattern_like(cluster_infos):
    """
    更保守的花紋判斷：
    只抓真正多色 / 條紋 / 印花，
    盡量排除：
    - 單色衣物陰影
    - 牛仔洗色
    - 光影造成的深淺變化
    """
    if len(cluster_infos) < 2:
        return False

    meaningful_clusters = []
    for info in cluster_infos:
        ratio = float(info["ratio"])
        if ratio < 0.12:
            continue
        meaningful_clusters.append(info)

    if len(meaningful_clusters) < 2:
        return False

    primary = meaningful_clusters[0]
    p_h, p_s, p_v = [float(x) for x in primary["hsv"]]
    p_ratio = float(primary["ratio"])

    def is_near_neutral(h, s, v):
        return s < 35 or v < 55 or v > 225

    def same_family(h1, s1, v1, h2, s2, v2):
        hue_gap = _hue_distance(h1, h2)
        sv_gap = abs(s1 - s2) + abs(v1 - v2)

        # 同色系深淺變化：像純色裙、外套、牛仔洗色
        if hue_gap < 18:
            return True

        # 藍色 / 牛仔常見洗色，不當成花紋
        if 160 <= h1 <= 250 and 160 <= h2 <= 250 and hue_gap < 28:
            return True

        # 低彩度亮部 / 陰影，不當成花紋
        if sv_gap < 70 and hue_gap < 28:
            return True

        return False

    strong_pattern_count = 0
    light_stripe_count = 0

    for secondary in meaningful_clusters[1:]:
        s_h, s_s, s_v = [float(x) for x in secondary["hsv"]]
        s_ratio = float(secondary["ratio"])

        hue_gap = _hue_distance(p_h, s_h)
        rgb_gap = _rgb_distance(primary["rgb"], secondary["rgb"])
        sv_gap = abs(p_s - s_s) + abs(p_v - s_v)

        if same_family(p_h, p_s, p_v, s_h, s_s, s_v):
            continue

        # 白條紋 / 淺色條紋
        is_secondary_light = s_v > 190 and s_s < 38
        is_primary_colored = p_s > 42 and 45 < p_v < 190

        if p_ratio >= 0.32 and s_ratio >= 0.18 and is_secondary_light and is_primary_colored:
            light_stripe_count += 1
            continue

        # 真正多色 / 印花
        if p_ratio >= 0.26 and s_ratio >= 0.18:
            if hue_gap >= 28 or rgb_gap >= 68 or sv_gap >= 105:
                strong_pattern_count += 1

    if strong_pattern_count >= 1:
        return True

    if light_stripe_count >= 1:
        return True

    # 第三群以上也要夠強，且不能都只是同色深淺
    if len(meaningful_clusters) >= 3:
        top3 = meaningful_clusters[:3]
        valid_groups = []

        for info in top3:
            h, s, v = [float(x) for x in info["hsv"]]
            if is_near_neutral(h, s, v):
                continue
            valid_groups.append(info)

        if len(valid_groups) >= 2:
            hue_gaps = []
            for i in range(len(valid_groups)):
                for j in range(i + 1, len(valid_groups)):
                    h1 = float(valid_groups[i]["hsv"][0])
                    h2 = float(valid_groups[j]["hsv"][0])
                    hue_gaps.append(_hue_distance(h1, h2))

            if any(gap >= 26 for gap in hue_gaps):
                ratios = [float(x["ratio"]) for x in valid_groups]
                if sum(ratios[:2]) >= 0.42:
                    return True

    return False


def extract_color(image: Image.Image):
    img = image.resize((180, 180)).convert("RGB")
    rgb = np.array(img).astype(np.float32)

    h, w, _ = rgb.shape
    weights_2d = _center_weight_mask(h, w)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    diff_rgb = max_rgb - min_rgb

    # 排除背景 / 白邊
    background_like = (
        ((max_rgb > 235) & (diff_rgb < 18)) |
        ((r > 190) & (g > 190) & (b > 190) & (diff_rgb < 20))
    )

    valid_mask = ~background_like

    if np.sum(valid_mask) < 300:
        valid_mask = np.ones((h, w), dtype=bool)

    flat_rgb = rgb[valid_mask]
    flat_weights = weights_2d[valid_mask].astype(np.float32)

    centers, labels = _weighted_kmeans(flat_rgb, flat_weights, k=4, max_iter=12)
    dominant_info, cluster_infos = _choose_dominant_cluster(centers, labels, flat_weights)

    if dominant_info is None:
        dominant_rgb = np.average(flat_rgb, axis=0, weights=flat_weights)
        dominant_hsv = _rgb_to_hsv_np(np.array([dominant_rgb], dtype=np.float32))[0]
        return map_color(rgb=dominant_rgb, hsv=dominant_hsv)

    # 先判斷是否屬於花紋 / 多色
    if _is_pattern_like(cluster_infos):
        return "花紋圖案"

    dominant_rgb = dominant_info["rgb"]
    dominant_hsv = dominant_info["hsv"]

    return map_color(rgb=dominant_rgb, hsv=dominant_hsv)