from models.clip_model import predict_topk, score_texts


VALID_LABELS = [
    "a clean product photo of a clothing item",
    "a clean product photo of a fashion item",
    "a photo of a person wearing clothes",
    "a product image of pants",
    "a product image of a skirt",
    "a product image of a dress",
    "a product image of a jacket",
    "a product image of a shirt",
    "a product image of shoes",
    "a product image of a hat",
]

INVALID_LABELS = [
    "a landscape photo",
    "a food photo",
    "a pet photo",
    "a photo of furniture",
    "a photo of electronics",
    "a photo of a car",
    "a screenshot of a website",
    "a poster or graphic design image",
    "a document or paper",
    "a close-up face photo",
]

PRODUCT_ROUTE_LABELS = [
    "a clean product photo of a single clothing item on a plain background",
    "a catalog photo of a single clothing item",
    "a studio product photo of a single fashion item",
    "a front-view product image of a clothing item",
    "a single garment centered on a plain background",
]

OUTFIT_ROUTE_LABELS = [
    "a person wearing clothes in a full-body outfit photo",
    "a person wearing clothes in a street outfit photo",
    "a person wearing a top or bottom in a lifestyle photo",
    "a fashion street snap with a person",
    "a person standing and wearing an outfit",
]


def _score_label_group(image, labels, topk=5):
    results = predict_topk(image, labels, topk=min(topk, len(labels)))
    best = results[0]
    return {
        "best_label": best["label"],
        "best_score": float(best["score"]),
        "results": results,
    }


def validate_fashion_input(image):
    valid_result = _score_label_group(image, VALID_LABELS, topk=5)
    invalid_result = _score_label_group(image, INVALID_LABELS, topk=5)

    valid_max = float(valid_result["best_score"])
    invalid_max = float(invalid_result["best_score"])
    margin = valid_max - invalid_max

    is_valid = (
        valid_max >= 0.045 and
        (
            margin >= 0.002 or
            valid_max >= invalid_max * 1.02
        )
    )

    top_matches = sorted(
        valid_result["results"] + invalid_result["results"],
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    return {
        "is_valid": bool(is_valid),
        "best_label": valid_result["best_label"] if valid_max >= invalid_max else invalid_result["best_label"],
        "best_score": max(valid_max, invalid_max),
        "valid_score": valid_max,
        "invalid_score": invalid_max,
        "margin": float(margin),
        "top_matches": top_matches,
    }


def detect_image_route(image):
    product_result = _score_label_group(image, PRODUCT_ROUTE_LABELS, topk=5)
    outfit_result = _score_label_group(image, OUTFIT_ROUTE_LABELS, topk=5)

    product_score = float(product_result["best_score"])
    outfit_score = float(outfit_result["best_score"])

    # 商品圖優先策略：
    # 只要 product 不明顯輸，就先當 product。
    # 因為目前商品圖走 detection 比較容易變差。
    if product_score >= outfit_score - 0.01:
        route = "product"
        best_label = product_result["best_label"]
        best_score = product_score
    else:
        route = "outfit"
        best_label = outfit_result["best_label"]
        best_score = outfit_score

    return {
        "route": route,
        "score": float(best_score),
        "product_score": product_score,
        "outfit_score": outfit_score,
        "best_label": best_label,
        "top_matches": sorted(
            product_result["results"] + outfit_result["results"],
            key=lambda x: x["score"],
            reverse=True
        )[:5]
    }

def detect_coarse_fashion_type(image):
    prompt_groups = {
        "pants": [
            "a product photo of pants",
            "a product photo of trousers",
            "a lower-body garment with two separate pant legs",
            "wide-leg pants laid flat",
            "jeans or trousers product photo",
        ],
        "skirt": [
            "a product photo of a skirt",
            "a lower-body garment without separated legs",
            "a straight skirt product photo",
            "a midi skirt laid flat",
            "a long skirt product photo",
        ],
        "dress": [
            "a product photo of a dress",
            "a one-piece dress garment",
            "a dress with a connected top and skirt",
            "a midi dress product photo",
            "a long dress laid flat",
        ],
        "upper_body": [
            "a product photo of a top",
            "an upper-body garment product photo",
            "a shirt, blouse, t-shirt, hoodie, sweater, or jacket",
            "a top clothing item laid flat",
            "a product image of upper-body clothing",
        ],
        "headwear": [
            "a product photo of a hat",
            "a cap or beanie product image",
            "a headwear item laid flat",
            "a fashion hat product photo",
            "a product image of headwear",
        ],
        "shoes": [
            "a product photo of shoes",
            "a pair of sneakers, boots, sandals, or heels",
            "footwear product image",
            "a pair of shoes laid flat",
            "a fashion footwear product photo",
        ],
    }

    flat_prompts = []
    prompt_to_group = {}

    for group_key, prompts in prompt_groups.items():
        for prompt in prompts:
            flat_prompts.append(prompt)
            prompt_to_group[prompt] = group_key

    results = score_texts(image, flat_prompts)

    grouped_scores = {key: [] for key in prompt_groups.keys()}
    for item in results:
        prompt = item["label"]
        score = float(item["score"])
        group_key = prompt_to_group[prompt]
        grouped_scores[group_key].append(score)

    score_map = {}
    for group_key, scores in grouped_scores.items():
        scores = sorted(scores, reverse=True)
        top_scores = scores[:2] if len(scores) >= 2 else scores
        score_map[group_key] = float(sum(top_scores) / len(top_scores))

    best_key = max(score_map, key=score_map.get)

    top_matches = sorted(
        [
            {
                "coarse_type": prompt_to_group[item["label"]],
                "prompt": item["label"],
                "score": float(item["score"]),
            }
            for item in results
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:8]

    return {
        "coarse_type": best_key,
        "score": score_map[best_key],
        "scores": score_map,
        "top_matches": top_matches,
    }