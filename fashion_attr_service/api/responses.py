from __future__ import annotations

from typing import Any


def build_success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
    }



def build_error_response(*, code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    error_payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "details": details,
    }
    return {
        "ok": False,
        "error": error_payload,
    }
