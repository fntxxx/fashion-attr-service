from __future__ import annotations

from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from fashion_attr_service.api.constants import (
    ERROR_CODE_INTERNAL_SERVER,
    ERROR_CODE_UNAUTHORIZED,
    ERROR_MESSAGE_INTERNAL_SERVER,
    ERROR_MESSAGE_UNAUTHORIZED,
)
from fashion_attr_service.api.exceptions import ApiErrorException
from fashion_attr_service.core.config import get_internal_api_token


_bearer_scheme = HTTPBearer(auto_error=False)


def require_internal_api_token(
    raw_authorization: str | None = Header(default=None, alias="Authorization", include_in_schema=False),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    configured_token = get_internal_api_token()
    if not configured_token:
        raise ApiErrorException(
            code=ERROR_CODE_INTERNAL_SERVER,
            message=ERROR_MESSAGE_INTERNAL_SERVER,
            status_code=500,
            details={"reason": "internal_api_token_not_configured"},
        )

    if raw_authorization is None:
        raise ApiErrorException(
            code=ERROR_CODE_UNAUTHORIZED,
            message=ERROR_MESSAGE_UNAUTHORIZED,
            status_code=401,
            details={"reason": "missing_authorization_header"},
        )

    if credentials is None:
        raise ApiErrorException(
            code=ERROR_CODE_UNAUTHORIZED,
            message=ERROR_MESSAGE_UNAUTHORIZED,
            status_code=401,
            details={"reason": "invalid_authorization_scheme"},
        )

    provided_token = credentials.credentials.strip()
    if not provided_token or provided_token != configured_token:
        raise ApiErrorException(
            code=ERROR_CODE_UNAUTHORIZED,
            message=ERROR_MESSAGE_UNAUTHORIZED,
            status_code=401,
            details={"reason": "invalid_api_token"},
        )
