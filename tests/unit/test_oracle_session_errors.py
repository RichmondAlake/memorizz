"""Oracle sessions that break: large entity writes, and pooled connections
whose session the server ended (ORA-03146, ORA-03138, DPY-4011)."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import oracledb
import pytest

from memorizz.memory_provider.oracle import OracleConfig, OracleProvider
from memorizz.memory_provider.oracle.provider import _GuardedPool, _PooledConnection


class _Connection:
    def __init__(self, *, alive=True, close_error=None):
        self.alive = alive
        self.close_error = close_error
        self.closed = self.pinged = False
        self.outputtypehandler = None

    def ping(self):
        self.pinged = True
        if not self.alive:
            raise oracledb.DatabaseError("DPY-4011: the database closed the connection")

    def is_healthy(self):
        return self.alive

    def close(self):
        if self.close_error:
            raise self.close_error
        self.closed = True


class _Pool:
    def __init__(self, connection):
        self.connection = connection
        self.dropped = []
        self.opened = 1

    def acquire(self):
        return self.connection

    def drop(self, connection):
        self.dropped.append(connection)


def _use(pool, error=None):
    with pool.acquire() as connection:
        connection.outputtypehandler = "handler"
        if error:
            raise error


@pytest.mark.unit
def test_a_connection_goes_back_to_the_pool_after_ordinary_use_and_errors():
    raw = _Connection()
    pool = _GuardedPool(_Pool(raw))
    _use(pool)
    assert raw.closed and not raw.pinged and raw.outputtypehandler == "handler"
    assert pool.opened == 1

    raw = _Connection()
    with pytest.raises(oracledb.DatabaseError):
        _use(_GuardedPool(_Pool(raw)), oracledb.DatabaseError("ORA-00942: no table"))
    assert raw.closed and not raw.pinged


@pytest.mark.unit
def test_a_live_session_that_sent_a_malformed_packet_leaves_the_pool():
    raw = _Connection()
    inner = _Pool(raw)
    with pytest.raises(oracledb.DatabaseError, match="ORA-03146"):
        _use(_GuardedPool(inner), oracledb.DatabaseError("ORA-03146: invalid buffer"))
    assert raw.pinged and inner.dropped == [raw] and not raw.closed


@pytest.mark.unit
def test_an_ended_session_is_closed_not_dropped():
    # pool.drop() of an ended session stops python-oracledb 26's pool thread.
    raw = _Connection(alive=False)
    inner = _Pool(raw)
    with pytest.raises(oracledb.DatabaseError, match="ORA-03138"):
        _use(_GuardedPool(inner), oracledb.DatabaseError("ORA-03138: terminated"))
    assert raw.pinged and raw.closed and inner.dropped == []


@pytest.mark.unit
def test_a_failed_release_never_hides_the_error_that_ended_the_block():
    raw = _Connection(close_error=oracledb.DatabaseError("DPY-4011: closed"))
    with pytest.raises(oracledb.DatabaseError, match="ORA-00942"):
        _use(_GuardedPool(_Pool(raw)), oracledb.DatabaseError("ORA-00942: no table"))

    raw = _Connection(close_error=oracledb.DatabaseError("DPY-4011: closed"))
    with pytest.raises(oracledb.DatabaseError, match="DPY-4011"):
        _use(_GuardedPool(_Pool(raw)))


@pytest.mark.unit
def test_closing_twice_releases_once():
    raw = _Connection()
    connection = _PooledConnection(_Pool(raw), raw)
    connection.close()
    raw.close_error = oracledb.DatabaseError("DPY-1001: not connected")
    connection.close()
    assert raw.closed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timeout", "expected"),
    [
        (60.0, {"getmode": oracledb.POOL_GETMODE_TIMEDWAIT, "wait_timeout": 60000}),
        (None, {}),
    ],
)
def test_the_pool_raises_instead_of_waiting_for_good(monkeypatch, timeout, expected):
    captured = {}

    def create_pool(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after the pool options")

    monkeypatch.setattr(oracledb, "create_pool", create_pool)
    config = OracleConfig(user="u", password="p", dsn="d", pool_wait_timeout=timeout)
    with pytest.raises(RuntimeError, match="stop after"):
        OracleProvider(config)
    assert {k: captured[k] for k in expected} == expected
    assert ("getmode" in captured) == bool(expected)


def _vector_provider(cursor):
    provider = OracleProvider.__new__(OracleProvider)
    provider._get_table_name = lambda _: "TEST.KNOWLEDGE_BASE"
    provider._set_vector_input_size = lambda *args: None
    connection = MagicMock()
    connection.cursor.return_value = cursor

    @contextmanager
    def connect():
        yield connection

    provider._get_connection = connect
    return provider, connection


@pytest.mark.unit
def test_a_new_vector_is_inserted_after_an_update_finds_nothing():
    cursor = MagicMock()
    cursor.rowcount = 0
    provider, connection = _vector_provider(cursor)
    provider.upsert_vector("ns", "source", [1, 0], metadata={"page": "p" * 20000})
    statements = [c.args[0].split()[0] for c in cursor.execute.call_args_list]
    assert statements == ["UPDATE", "INSERT"]
    assert connection.commit.called


@pytest.mark.unit
def test_a_vector_id_held_by_an_unrelated_row_is_refused():
    cursor = MagicMock()
    cursor.rowcount = 0

    def execute(sql, params):
        if sql.split()[0] == "INSERT":
            raise RuntimeError("ORA-00001: unique constraint violated")

    cursor.execute.side_effect = execute
    provider, connection = _vector_provider(cursor)
    with pytest.raises(ValueError, match="collides"):
        provider.upsert_vector("ns", "source", [1, 0], metadata={})
    statements = [c.args[0].split()[0] for c in cursor.execute.call_args_list]
    assert statements == ["UPDATE", "INSERT", "UPDATE"]
    assert not connection.commit.called
