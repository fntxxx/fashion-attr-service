from models.clip_model import predict

categories = [
    "t-shirt",
    "shirt",
    "hoodie",
    "sweater",
    "jacket",
    "coat",
    "dress",
    "pants",
    "skirt",
    "shoes"
]


def classify_category(image):
    label, score = predict(image, categories)
    return label, score