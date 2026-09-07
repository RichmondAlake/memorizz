"""Atomic coordination on local files and explicitly isolated database targets."""

import multiprocessing
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from memorizz.coordination.shared_memory.shared_memory import SharedMemory
from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def filesystem_provider(path):
    return FileSystemProvider(FileSystemConfig(root_path=path, use_faiss=False))


def append_in_process(path, memory_id, worker, ready, start, results):
    provider = filesystem_provider(path)
    ready.put(worker)
    assert start.wait(10)
    shared = SharedMemory(provider)
    results.put(
        all(
            shared.add_blackboard_entry(
                memory_id,
                str(worker),
                {"number": number},
                "result",
                entry_id=f"{worker}-{number}",
            )
            for number in range(10)
        )
    )


def test_filesystem_writers_in_separate_processes_preserve_every_entry(tmp_path):
    provider = filesystem_provider(tmp_path)
    shared = SharedMemory(provider)
    memory_id = shared.create_shared_session("root", user_id="alice")
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    workers = [
        context.Process(
            target=append_in_process,
            args=(str(tmp_path), memory_id, number, ready, start, results),
        )
        for number in range(4)
    ]
    try:
        for worker in workers:
            worker.start()
        assert {ready.get(timeout=15) for _ in workers} == set(range(4))
        start.set()
        assert all(results.get(timeout=20) for _ in workers)
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        entries = SharedMemory(filesystem_provider(tmp_path)).get_blackboard_entries(
            memory_id
        )
        assert len(entries) == len({entry["memory_id"] for entry in entries}) == 40
    finally:
        start.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)
        ready.close()
        results.close()


@pytest.fixture(params=["filesystem", "mongodb", "oracle"])
def atomic_provider(request, tmp_path):
    if request.param == "filesystem":
        yield filesystem_provider(tmp_path)
        return
    if os.getenv("MEMORIZZ_DELEGATION_LIVE") != "1":
        pytest.skip("Live delegation requires explicitly supplied disposable databases")
    if request.param == "mongodb":
        from memorizz.memory_provider.mongodb.provider import (
            MongoDBConfig,
            MongoDBProvider,
        )

        uri = os.getenv("MEMORIZZ_OBS_TEST_MONGODB_URI")
        if not uri:
            pytest.fail("Explicit MEMORIZZ_OBS_TEST_MONGODB_URI is required")
        name = "memorizz_delegation_test_" + uuid.uuid4().hex
        provider = MongoDBProvider(
            MongoDBConfig(uri=uri, db_name=name, lazy_vector_indexes=True)
        )
        try:
            yield provider
        finally:
            provider.client.drop_database(name)
            provider.close()
        return

    import oracledb

    from memorizz.memory_provider.oracle.provider import OracleProvider

    user = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_USER", "")
    dsn = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_DSN")
    password = os.getenv("MEMORIZZ_OBS_TEST_ORACLE_PASSWORD")
    if (
        not re.fullmatch(r"MEMORIZZ_OBS_TEST_[A-Z0-9_]{1,80}", user.upper())
        or not dsn
        or not password
    ):
        pytest.fail(
            "Explicit dedicated MEMORIZZ_OBS_TEST_* Oracle credentials are required"
        )
    pool = oracledb.create_pool(
        user=user, password=password, dsn=dsn, min=1, max=6, increment=1
    )
    table = "MZ_DELEGATE_" + uuid.uuid4().hex[:16].upper()

    @contextmanager
    def connection():
        with pool.acquire() as conn:
            yield conn

    # Exercise the production shared-memory storage, raw reads and CLOB CAS.
    # Do not initialize/migrate unrelated provider tables in a host schema.
    provider = object.__new__(OracleProvider)
    provider._get_connection = connection
    provider._get_table_name = lambda memory_type: table
    provider.store = lambda data, memory_type: provider._store_shared_memory(data)
    created = False
    try:
        with connection() as conn:
            conn.cursor().execute(
                f"""CREATE TABLE {table} (
                id RAW(16) PRIMARY KEY, memory_id VARCHAR2(255) UNIQUE,
                content CLOB, memory_type VARCHAR2(64), scope VARCHAR2(32),
                owner_agent_id VARCHAR2(255), embedding CLOB, access_list CLOB,
                created_at TIMESTAMP, updated_at TIMESTAMP)"""
            )
            created = True
        yield provider
    finally:
        if created:
            with connection() as conn:
                conn.cursor().execute(f"DROP TABLE {table} PURGE")
        pool.close(force=True)


def test_conditional_write_uses_exact_snapshot_and_supports_large_payloads(
    atomic_provider,
):
    provider = atomic_provider
    shared = SharedMemory(provider)
    memory_id = shared.create_shared_session("root")
    original = provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)["content"]
    assert isinstance(original, str)
    assert shared.add_blackboard_entry(memory_id, "doc", "📘" * 40_000, "result")
    assert not provider.compare_and_swap_shared_memory(
        memory_id, original, "stale writer"
    )
    assert shared.add_blackboard_entry(memory_id, "slides", "saved", "result")
    entries = shared.get_blackboard_entries(memory_id)
    assert len(entries) == 2 and entries[0]["content"] == "📘" * 40_000


def test_concurrent_appends_status_and_registration_preserve_each_other(
    atomic_provider,
):
    shared = SharedMemory(atomic_provider)
    memory_id = shared.create_shared_session("root", user_id="alice")
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = [
            pool.submit(
                shared.add_blackboard_entry,
                memory_id,
                "doc",
                number,
                "result",
                entry_id=f"receipt-{number}",
            )
            for number in range(30)
        ]
        pending.append(
            pool.submit(shared.register_sub_agents, memory_id, "root", ["child"])
        )
        pending.append(
            pool.submit(
                shared.update_session_status,
                memory_id,
                "partial",
                outcome="partial",
                counts={"completed": 1, "failed": 1},
            )
        )
        assert all(future.result(timeout=30) for future in pending)
    payload = shared._decode_payload(
        atomic_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
    )
    results = [
        entry for entry in payload["blackboard"] if entry["entry_type"] == "result"
    ]
    assert len(results) == len({entry["memory_id"] for entry in results}) == 30
    assert payload["status"] == payload["outcome"] == "partial"
    assert payload["counts"] == {"completed": 1, "failed": 1}
    assert payload["sub_agent_ids"] == ["child"]
    assert payload["revision"] == 32
