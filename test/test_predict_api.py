from __future__ import annotations

import importlib
import io
import os
import sys
import types
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image


def _install_stub_predict_pipeline_module() -> None:
    module_name = "fashion_attr_service.services.predict_pipeline"
    stub_module = types.ModuleType(module_name)

    def _default_warmup():
        return {
            "ok": True,
            "data": {
                "service": "fashion-attr-service",
                "model": {
                    "backend": "marqo_fashionsiglip",
                    "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                },
                "warmup": {
                    "validation_best_label": "t-shirt",
                    "coarse_type": "top",
                    "category": "t_shirt",
                    "color": "neutral_gray",
                },
            },
        }

    def _default_predict_attributes(_image, model_backend=None, *, include_debug=False):
        raise AssertionError("predict_attributes stub should be patched by each test")

    stub_module.run_warmup = _default_warmup
    stub_module.predict_attributes = _default_predict_attributes
    sys.modules[module_name] = stub_module


_install_stub_predict_pipeline_module()
main_module = importlib.import_module("fashion_attr_service.main")
app = main_module.app


class PredictApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        @app.get("/__test/http-exception")
        def raise_http_exception_for_test():
            raise HTTPException(status_code=409, detail="conflict")

    def setUp(self) -> None:
        self._original_token = os.environ.get("INTERNAL_API_TOKEN")
        os.environ["INTERNAL_API_TOKEN"] = "test-token"
        self.client = TestClient(app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        if self._original_token is None:
            os.environ.pop("INTERNAL_API_TOKEN", None)
        else:
            os.environ["INTERNAL_API_TOKEN"] = self._original_token

    @staticmethod
    def _auth_headers(token: str = "test-token") -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _build_image_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 255, 255)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_predict_returns_200_with_unified_success_envelope(self) -> None:
        success_payload = {
            "ok": True,
            "data": {
                "route": "product",
                "coarseType": "top",
                "name": "T 恤",
                "category": "top",
                "categoryLabel": "上衣",
                "color": "light_beige",
                "colorLabel": "淺米白",
                "occasion": ["campus_casual"],
                "season": ["spring"],
                "score": 0.91,
                "scores": {
                    "mainCategory": 0.94,
                    "category": 0.91,
                    "occasion": 0.88,
                    "color": 0.92,
                    "season": 0.82,
                },
                "candidates": {
                    "category": [{"value": "top", "label": "上衣", "score": 0.91}],
                    "color": [{"value": "light_beige", "label": "淺米白", "score": 0.92}],
                    "occasion": [{"value": "campus_casual", "label": "校園休閒", "score": 0.88}],
                    "season": [{"value": "spring", "label": "春季", "score": 0.82}],
                },
                "detected": False,
                "detectedLabel": None,
                "bbox": None,
                "validation": {
                    "best_label": "t-shirt",
                    "valid_score": 0.93,
                    "invalid_score": 0.07,
                },
            },
        }

        with patch.object(main_module, "predict_attributes", return_value=success_payload):
            response = self.client.post(
                "/predict",
                files={"image": ("valid.png", self._build_image_bytes(), "image/png")},
                headers=self._auth_headers(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), success_payload)


    def test_predict_rejects_json_request_without_image_upload(self) -> None:
        response = self.client.post(
            "/predict",
            json={"base64": "iVBORw0KGgoAAAANSUhEUgAA..."},
            headers=self._auth_headers(),
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "request_validation_error",
                    "message": "請求參數驗證失敗。",
                    "details": {
                        "fields": [
                            {
                                "loc": ["body", "image"],
                                "msg": "Field required",
                                "type": "missing",
                            }
                        ]
                    },
                },
            },
        )


    def test_predict_returns_400_with_unified_business_error_envelope(self) -> None:
        from fashion_attr_service.api.exceptions import PredictRejectedError

        with patch.object(
            main_module,
            "predict_attributes",
            side_effect=PredictRejectedError(
                reason="not_fashion_image",
                validation={
                    "best_label": "person",
                    "valid_score": 0.18,
                    "invalid_score": 0.82,
                },
                status_code=400,
            ),
        ):
            response = self.client.post(
                "/predict",
                files={"image": ("invalid.png", self._build_image_bytes(), "image/png")},
                headers=self._auth_headers(),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "predict_rejected",
                    "message": "輸入圖片未通過服飾商品圖驗證。",
                    "details": {
                        "reason": "not_fashion_image",
                        "validation": {
                            "best_label": "person",
                            "valid_score": 0.18,
                            "invalid_score": 0.82,
                        },
                    },
                },
            },
        )

    def test_predict_returns_422_with_unified_validation_error_envelope(self) -> None:
        response = self.client.post("/predict", files={}, headers=self._auth_headers())

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["ok"], False)
        self.assertEqual(body["error"]["code"], "request_validation_error")
        self.assertEqual(body["error"]["message"], "請求參數驗證失敗。")
        self.assertIn("fields", body["error"]["details"])
        self.assertTrue(body["error"]["details"]["fields"])

    def test_http_exception_uses_unified_error_envelope(self) -> None:
        response = self.client.get("/__test/http-exception")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "http_exception",
                    "message": "conflict",
                    "details": {
                        "detail": "conflict",
                    },
                },
            },
        )

    def test_unexpected_exception_uses_unified_error_envelope(self) -> None:
        with patch.object(main_module, "predict_attributes", side_effect=RuntimeError("boom")):
            response = self.client.post(
                "/predict",
                files={"image": ("valid.png", self._build_image_bytes(), "image/png")},
                headers=self._auth_headers(),
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "internal_server_error",
                    "message": "伺服器發生未預期錯誤。",
                    "details": None,
                },
            },
        )


    def test_predict_openapi_request_body_only_accepts_multipart_image_upload(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        request_body = response.json()["paths"]["/predict"]["post"]["requestBody"]
        self.assertEqual(set(request_body["content"].keys()), {"multipart/form-data"})
        multipart_schema = request_body["content"]["multipart/form-data"]["schema"]

        self.assertEqual(multipart_schema["type"], "object")
        self.assertEqual(multipart_schema["required"], ["image"])
        self.assertEqual(multipart_schema["properties"]["image"]["format"], "binary")

    def test_warmup_returns_500_with_unified_error_envelope_when_warmup_fails(self) -> None:
        with patch.object(
            main_module,
            "run_warmup",
            return_value={
                "ok": False,
                "service": "fashion-attr-service",
                "model": {
                    "backend": "marqo_fashionsiglip",
                    "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                },
                "error": "model initialization failed",
            },
        ):
            response = self.client.get("/warmup", headers=self._auth_headers())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "warmup_failed",
                    "message": "模型 warmup 失敗。",
                    "details": {
                        "service": "fashion-attr-service",
                        "model": {
                            "backend": "marqo_fashionsiglip",
                            "model_name": "hf-hub:Marqo/marqo-fashionSigLIP",
                        },
                        "reason": "model initialization failed",
                    },
                },
            },
        )

    def test_predict_returns_401_when_token_is_missing(self) -> None:
        response = self.client.post(
            "/predict",
            files={"image": ("valid.png", self._build_image_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.json(),
            {
                "ok": False,
                "error": {
                    "code": "unauthorized",
                    "message": "缺少或無效的 API Token。",
                    "details": {"reason": "missing_authorization_header"},
                },
            },
        )

    def test_predict_returns_401_when_bearer_format_is_invalid(self) -> None:
        response = self.client.post(
            "/predict",
            files={"image": ("valid.png", self._build_image_bytes(), "image/png")},
            headers={"Authorization": "Token test-token"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["details"]["reason"], "invalid_authorization_scheme")

    def test_predict_returns_401_when_token_is_incorrect(self) -> None:
        response = self.client.post(
            "/predict",
            files={"image": ("valid.png", self._build_image_bytes(), "image/png")},
            headers=self._auth_headers("wrong-token"),
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["details"]["reason"], "invalid_api_token")

    def test_health_remains_public_without_token(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_openapi_remains_public_without_token(self) -> None:
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/predict", response.json()["paths"])


if __name__ == "__main__":
    unittest.main()
