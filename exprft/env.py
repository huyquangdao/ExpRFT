from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def require(name: str) -> str:
    load_dotenv()
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set: export it or add '{name}=...' to {ENV_FILE} (see .env.example)")
    return value
