from __future__ import annotations

import uvicorn

from turboquant.studio_api import create_app


def main() -> int:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
