"""Shared test setup.

Unit tests need nothing running. Tests marked `db` use the local Postgres and
are skipped, not failed, when it is down:
    pytest              # everything
    pytest -m "not db"  # unit tests only
"""

import asyncio

import pytest
from sqlalchemy import text

from app.db import SessionLocal, dispose_engine


def _database_up() -> bool:
    async def ping() -> None:
        try:
            async with SessionLocal() as session:
                await asyncio.wait_for(session.execute(text("select 1")), timeout=5)
        finally:
            # Each test runs its own event loop; pooled connections must not
            # outlive the loop that opened them.
            await dispose_engine()

    try:
        asyncio.run(ping())
    except Exception:  # noqa: BLE001 -- any failure means "not available"
        return False
    return True


@pytest.fixture(scope="session")
def database_up() -> bool:
    return _database_up()


@pytest.fixture(autouse=True)
def _skip_without_database(request, database_up) -> None:
    if request.node.get_closest_marker("db") and not database_up:
        pytest.skip("Postgres is not running (docker start fmep-pg)")
