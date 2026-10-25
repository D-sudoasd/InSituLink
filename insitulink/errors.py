"""Input requests shared by the deterministic pipeline stages."""

from __future__ import annotations


class InputError(ValueError):
    """An experiment input must be supplied or corrected before continuing."""

    def __init__(self, message: str, code: str = "invalid_input", **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), **self.details}
