"""Run the API locally on Windows.

`uvicorn app.main:app` fails here: uvicorn installs a ProactorEventLoop of its
own, and psycopg cannot run async on that loop. Setting the policy beforehand
does not help, because uvicorn replaces the loop afterwards. So the loop is made
here and uvicorn is told not to make one (loop="none").

On Linux, and in the Docker image, plain uvicorn is fine and this script is not
used.

    python scripts/dev_api.py            # http://localhost:8000
    python scripts/dev_api.py --port 8080
"""

import argparse
import asyncio
import selectors
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402  -- after sys.path so `app` is importable


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the API for local development.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart when code changes")
    args = parser.parse_args()

    if args.reload:
        # Reload needs uvicorn's own supervisor, which spawns workers that make
        # their own loops -- so the selector policy has to be set in each worker.
        # app/__init__.py already does that on import, which is enough here.
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
        return

    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    server = uvicorn.Server(
        uvicorn.Config("app.main:app", host=args.host, port=args.port, loop="none")
    )
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    main()
