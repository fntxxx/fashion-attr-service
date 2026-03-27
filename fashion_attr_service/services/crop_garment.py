from PIL import Image


def crop_image_by_bbox(
    image: Image.Image,
    bbox,
    padding_ratio: float = 0.08
) -> Image.Image:
    if not bbox:
        return image

    x1, y1, x2, y2 = bbox
    width, height = image.size

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image

    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))