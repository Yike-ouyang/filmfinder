from __future__ import annotations


class APIError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)
