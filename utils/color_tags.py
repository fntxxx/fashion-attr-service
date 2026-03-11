def color_tone_to_tags(color_tone: str):
    """
    後端基礎色系 -> 正式產品 UI 色票 tag
    先做穩定單層映射，之後若要支援多標籤，可在這裡擴充。
    """
    mapping = {
        "白色系": ["淺米白"],
        "黑色系": ["深灰黑"],
        "灰色系": ["中性灰"],
        "米色系": ["淺米白"],
        "卡其色系": ["大地棕"],
        "咖啡色系": ["大地棕"],
        "紅色系": ["暖橘紅"],
        "綠色系": ["自然綠"],
        "藍色系": ["清爽藍"],
        "紫色系": ["優雅紫"],
        "花紋圖案": ["花紋圖案"],
    }

    return mapping.get(color_tone, [])


def build_color_payload(color_tone: str):
    return {
        "colorTone": color_tone,
        "colorTags": color_tone_to_tags(color_tone),
    }