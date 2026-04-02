from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 只在本機開發時載入
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

INTERNAL_API_TOKEN_ENV_NAME = "INTERNAL_API_TOKEN"


def get_internal_api_token() -> str:
    return os.getenv(INTERNAL_API_TOKEN_ENV_NAME, "")