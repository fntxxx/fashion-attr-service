import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from utils.color_map import map_color


def extract_color(image: Image.Image):

    img = image.resize((150, 150))
    pixels = np.array(img).reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=5)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.argmax()]

    return map_color(dominant)