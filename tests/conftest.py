import sqlite3

import pytest

from fwsp import db


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    db.init_schema(conn)
    yield conn
    conn.close()
