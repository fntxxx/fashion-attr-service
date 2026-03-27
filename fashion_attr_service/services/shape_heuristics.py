import numpy as np
from PIL import Image


def _binarize_foreground(image: Image.Image):
    img = image.convert("RGB").resize((256, 256))
    arr = np.array(img).astype(np.int16)

    # 商品圖大多是灰底，先用與背景色差做簡單前景分離
    bg = arr[0:20, 0:20].reshape(-1, 3).mean(axis=0)
    diff = np.sqrt(((arr - bg) ** 2).sum(axis=2))

    mask = diff > 18
    return mask.astype(np.uint8)


def estimate_pants_vs_skirt(image: Image.Image):
    """
    回傳:
    - "pants": 比較像褲子
    - "skirt": 比較像裙子
    - "unknown": 不明顯
    """
    mask = _binarize_foreground(image)
    h, w = mask.shape

    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return "unknown"

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    obj = mask[y1:y2 + 1, x1:x2 + 1]
    oh, ow = obj.shape

    if oh < 40 or ow < 40:
        return "unknown"

    # 看下半部中央有沒有明顯分腿空隙
    lower_start = int(oh * 0.58)
    lower = obj[lower_start:, :]

    # 每一列的前景寬度與中間空白
    center_band = lower[:, int(ow * 0.42): int(ow * 0.58)]
    center_fill_ratio = center_band.mean()

    # 底部左右兩側是否各有一塊前景，中間較空
    bottom = obj[int(oh * 0.82):, :]
    left_fill = bottom[:, :int(ow * 0.35)].mean()
    center_fill = bottom[:, int(ow * 0.40):int(ow * 0.60)].mean()
    right_fill = bottom[:, int(ow * 0.65):].mean()

    # 褲子常見特徵：
    # 左右兩邊都有布料，但中間比較空
    pants_score = 0
    skirt_score = 0

    if left_fill > 0.35 and right_fill > 0.35 and center_fill < 0.22:
        pants_score += 2

    if center_fill_ratio < 0.22:
        pants_score += 1

    # 裙子常見特徵：
    # 下半部連續，不太有中間明顯分腿
    if center_fill > 0.28:
        skirt_score += 2

    if center_fill_ratio > 0.26:
        skirt_score += 1

    # 如果整體往下逐漸外擴，也偏裙子
    widths = []
    for ratio in [0.35, 0.50, 0.65, 0.80]:
        row = obj[int(oh * ratio)]
        xs_row = np.where(row > 0)[0]
        if len(xs_row) > 0:
            widths.append(xs_row.max() - xs_row.min() + 1)

    if len(widths) >= 3:
        if widths[-1] > widths[0] * 1.15:
            skirt_score += 1

    if pants_score >= skirt_score + 1:
        return "pants"
    if skirt_score >= pants_score + 1:
        return "skirt"

    return "unknown"