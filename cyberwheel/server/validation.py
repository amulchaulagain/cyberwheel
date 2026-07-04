"""Request validation helpers.

Endpoints take plain dicts (no pydantic models: the repo pins pydantic v1,
and keeping it out of handler signatures makes a future v2 migration a
non-event) and validate through these helpers.
"""

from __future__ import annotations

import re


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str = "bad_request"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


def require(condition, message: str, status: int = 400) -> None:
    if not condition:
        raise ApiError(status, message)


def not_found(message: str) -> ApiError:
    return ApiError(404, message, code="not_found")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    return slug[:48] or "run"
