from __future__ import annotations

import importlib
import sys
import types
import unittest

from fashion_attr_service.api.formatters import build_category_candidates, build_predict_payload
from fashion_attr_service.utils.category_catalog import (
    FINE_CATEGORY_DEFAULTS,
    FINE_CATEGORY_MAPS,
    MAIN_CATEGORY_LABEL_MAP,
    get_fine_category_label,
    get_main_category_label,
    normalize_category_key,
)


def _install_predict_pipeline_stubs() -> None:
    model_module = types.ModuleType("fashion_attr_service.models.fashion_siglip_model")
    model_module.BACKEND_SPEC = types.SimpleNamespace(key="stub_backend", model_name="stub_model")
    model_module.MODEL_BACKEND = "stub_backend"
    model_module.encode_image_feature = lambda *args, **kwargs: {"feature": "stub"}
    model_module.get_clip_model = lambda *args, **kwargs: object()
    sys.modules[model_module.__name__] = model_module

    attr_module = types.ModuleType("fashion_attr_service.services.attribute_heads")
    attr_module.infer_color = lambda *args, **kwargs: {"color": "light_beige", "heuristicTone": None, "candidates": [], "colorLabel": "淺米白"}
    attr_module.infer_occasions = lambda *args, **kwargs: {"selected": [], "candidates": []}
    attr_module.infer_seasons = lambda *args, **kwargs: {"selected": [], "candidates": []}
    sys.modules[attr_module.__name__] = attr_module

    classify_module = types.ModuleType("fashion_attr_service.services.classify_category")
    classify_module.classify_category = lambda *args, **kwargs: {"scores": {"category": 0.0}, "mainCategoryKey": "upper_body", "categoryKey": "shirt"}
    sys.modules[classify_module.__name__] = classify_module

    postprocess_module = types.ModuleType("fashion_attr_service.services.postprocess_category")
    postprocess_module.postprocess_category = lambda result, *args, **kwargs: result
    sys.modules[postprocess_module.__name__] = postprocess_module

    validate_module = types.ModuleType("fashion_attr_service.services.validate_input")
    validate_module.detect_coarse_fashion_type = lambda *args, **kwargs: {"coarse_type": "upper_body", "score": 0.0}
    validate_module.validate_fashion_input = lambda *args, **kwargs: {
        "is_valid": True,
        "best_label": "shirt",
        "valid_score": 1.0,
        "invalid_score": 0.0,
    }
    sys.modules[validate_module.__name__] = validate_module


_install_predict_pipeline_stubs()
predict_pipeline = importlib.import_module("fashion_attr_service.services.predict_pipeline")


class CategoryCatalogInvariantTests(unittest.TestCase):
    def test_category_catalog_defaults_are_valid_members(self) -> None:
        for main_key, default_key in FINE_CATEGORY_DEFAULTS.items():
            self.assertIn(default_key, FINE_CATEGORY_MAPS[main_key])
            self.assertEqual(get_main_category_label(main_key), MAIN_CATEGORY_LABEL_MAP[main_key])
            self.assertEqual(
                get_fine_category_label(main_key, default_key),
                FINE_CATEGORY_MAPS[main_key][default_key],
            )

    def test_normalize_category_key_falls_back_to_main_category_default(self) -> None:
        self.assertEqual(normalize_category_key("upper_body", "unknown_key"), "shirt")
        self.assertEqual(normalize_category_key("skirt", "unknown_key"), "midi_skirt")
        self.assertEqual(normalize_category_key("shoes", "boots"), "boots")


class FormatterInvariantTests(unittest.TestCase):
    def test_build_category_candidates_fallback_keeps_selected_value_as_only_positive_score(self) -> None:
        result = build_category_candidates(
            {
                "mainCategoryKey": "pants",
                "categoryKey": "trousers",
                "scores": {"mainCategory": 0.9, "category": 0.8},
                "candidateScoreMaps": {},
            }
        )

        positive = [item for item in result if item["score"] > 0]
        self.assertEqual(len(positive), 1)
        self.assertEqual(positive[0]["value"], "pants")
        self.assertEqual(positive[0]["score"], 1.0)

    def test_build_predict_payload_uses_top_candidate_scores_without_changing_contract(self) -> None:
        payload = build_predict_payload(
            route="product",
            coarse_type="upper_body",
            category_result={
                "mainCategoryKey": "upper_body",
                "categoryKey": "shirt",
                "mainCategory": "上身",
                "category": "襯衫",
                "scores": {"mainCategory": 0.7, "category": 0.8},
                "candidateScoreMaps": {
                    "mainCategory": {"upper_body": 1.0},
                    "category": {"shirt": 0.8},
                },
            },
            color_payload={
                "color": "light_beige",
                "colorLabel": "淺米白",
                "candidates": [{"value": "light_beige", "label": "淺米白", "score": 0.66}],
            },
            occasions={
                "selected": ["business_casual"],
                "candidates": [{"value": "business_casual", "label": "商務休閒", "score": 0.77}],
            },
            seasons={
                "selected": ["spring"],
                "candidates": [{"value": "spring", "label": "春季", "score": 0.55}],
            },
            validation={"best_label": "shirt", "valid_score": 0.9, "invalid_score": 0.1},
            detection={"detected": False, "label": None, "bbox": None},
            final_score=0.8,
        )

        self.assertEqual(payload["scores"]["occasion"], 0.77)
        self.assertEqual(payload["scores"]["color"], 0.66)
        self.assertEqual(payload["scores"]["season"], 0.55)
        self.assertEqual(payload["category"], "top")
        self.assertIn("candidates", payload)
        self.assertIn("validation", payload)


class PredictPipelineHelperTests(unittest.TestCase):
    def test_build_detection_from_color_payload_returns_default_when_detection_missing(self) -> None:
        self.assertEqual(
            predict_pipeline._build_detection_from_color_payload({}),
            predict_pipeline._build_default_detection(),
        )

    def test_build_detection_from_color_payload_extracts_detection_fields(self) -> None:
        detection = predict_pipeline._build_detection_from_color_payload(
            {
                "focusDebug": {
                    "detection": {
                        "detected": True,
                        "label": "shirt",
                        "mainCategoryKey": "upper_body",
                        "bbox": [1, 2, 3, 4],
                        "score": 0.42,
                    }
                }
            }
        )

        self.assertEqual(
            detection,
            {
                "detected": True,
                "label": "shirt",
                "mainCategoryKey": "upper_body",
                "bbox": [1, 2, 3, 4],
                "score": 0.42,
            },
        )

    def test_resolve_final_score_matches_existing_weighting_rule(self) -> None:
        self.assertEqual(predict_pipeline._resolve_final_score(0.8, {"detected": False, "score": 0.2}), 0.8)
        self.assertAlmostEqual(predict_pipeline._resolve_final_score(0.8, {"detected": True, "score": 0.2}), 0.65)


if __name__ == "__main__":
    unittest.main()
