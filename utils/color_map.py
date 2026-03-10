def map_color(rgb):

    r, g, b = rgb

    if r > 200 and g > 200 and b > 200:
        return "白色系"

    if r < 60 and g < 60 and b < 60:
        return "黑色系"

    if abs(r - g) < 20 and abs(g - b) < 20:
        return "灰色系"

    if r > 180 and g > 160 and b < 120:
        return "米色系"

    if r > g and r > b:
        return "紅色系"

    if g > r and g > b:
        return "綠色系"

    if b > r and b > g:
        return "藍色系"

    return "其他"