import numpy as np


def map_color(rgb, hsv=None):
    r, g, b = [float(x) for x in rgb]

    if hsv is None:
        hsv = np.array([0.0, 0.0, 0.0])

    h, s, v = [float(x) for x in hsv]

    # -----------------------
    # 黑 / 白 / 灰
    # -----------------------

    if v < 45:
        return "黑色系"

    if s < 18:
        if v > 225:
            return "白色系"
        if v > 95:
            return "灰色系"
        return "黑色系"

    # -----------------------
    # 米 / 卡其 / 咖啡
    # -----------------------

    if r > 190 and g > 175 and b > 135:
        return "米色系"

    if r > 150 and g > 120 and b < 120:
        return "卡其色系"

    if r > 95 and g > 65 and b < 80:
        return "咖啡色系"

    # -----------------------
    # Hue based
    # -----------------------

    # 紅
    if (h >= 345) or (h < 20):
        return "紅色系"

    # 橘
    if 20 <= h < 45:
        if r > 150 and g > 120:
            return "卡其色系"
        return "紅色系"

    # 黃
    if 45 <= h < 65:
        if s < 80:
            return "卡其色系"
        return "紅色系"

    # 綠
    if 65 <= h < 160:
        return "綠色系"

    # 藍
    if 160 <= h < 250:
        return "藍色系"

    # 紫
    if 250 <= h < 320:
        return "紫色系"

    # 粉 / 紅紫
    if 320 <= h < 345:
        return "紅色系"

    # -----------------------
    # fallback
    # -----------------------

    if b >= r and b >= g:
        return "藍色系"

    if g >= r and g >= b:
        return "綠色系"

    if r >= g and r >= b:
        return "紅色系"

    return "其他"