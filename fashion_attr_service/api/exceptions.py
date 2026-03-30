from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fashion_attr_service.api.constants import (
    ERROR_CODE_PREDICT_REJECTED,
    ERROR_MESSAGE_PREDICT_REJECTED,
)
from fashion_attr_service.api.responses import build_error_response


@dataclass
class ApiErrorException(Exception):
    code: str
    message: str
    status_code: int
    details: Any | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def payload(self) -> dict[str, Any]:
        return build_error_response(code=self.code, message=self.message, details=self.details)


class PredictRejectedError(ApiErrorException):
    def __init__(self, *, reason: str, validation: dict[str, Any], status_code: int = 400) -> None:
        super().__init__(
            code=ERROR_CODE_PREDICT_REJECTED,
            message=ERROR_MESSAGE_PREDICT_REJECTED,
            status_code=status_code,
            details={
                "reason": reason,
                "validation": validation,
            },
        )
