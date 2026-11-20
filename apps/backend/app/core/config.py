from __future__ import annotations

import os
from pathlib import Path


def artifacts_dir():
    value = os.environ.get("FILMFINDER_ARTIFACTS_DIR")
    if value:
        return Path(value).expanduser()
    return None
