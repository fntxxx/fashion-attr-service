def infer_style(main_category: str, fine_category: str) -> str:
    if fine_category in {
        "blazer", "shirt", "loafers", "heels", "trench_coat"
    }:
        return "formal"

    if fine_category in {
        "running_shoes", "leggings", "jogger_pants", "windbreaker", "hoodie"
    }:
        return "sport"

    return "casual"


def infer_season(main_category: str, fine_category: str) -> str:
    if fine_category in {
        "tank_top", "camisole", "shorts", "denim_shorts", "sandals", "slip_dress"
    }:
        return "summer"

    if fine_category in {
        "coat", "trench_coat", "puffer_jacket", "knit_sweater",
        "cardigan", "beanie", "boots", "ankle_boots"
    }:
        return "winter"

    return "spring_autumn"