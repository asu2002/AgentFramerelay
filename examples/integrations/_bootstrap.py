"""Make the local src-layout package importable when examples run from source."""

import sys
from pathlib import Path


def ensure_local_package() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
