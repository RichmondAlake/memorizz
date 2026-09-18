"""Opt-in live contracts. Every write uses an isolated, disposable test scope."""

import os
import uuid
from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from tests.unit.test_notion_provider import Embedder


def test_live_local_oracle_vector_composition(tmp_path):
    if os.getenv("MEMORIZZ_TEST_NOTION_ORACLE") != "1":
        pytest.skip(
            "Set MEMORIZZ_TEST_NOTION_ORACLE=1 for the scoped local Oracle test"
        )
    from dotenv import load_dotenv

    from memorizz.memory_provider import NotionConfig, NotionProvider
    from memorizz.memory_provider.notion import NotionClient
    from memorizz.memory_provider.oracle import OracleProvider
    from tests.fixtures.notion_api import NotionAPI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    dsn = os.getenv("ORACLE_DSN", "")
    if not dsn.startswith(("localhost:", "127.0.0.1:")):
        pytest.fail(
            "The live vector test only permits an explicitly configured loopback Oracle DSN"
        )

    class OracleEmbedder(Embedder):
        dimensions = 256

        def get_embedding(self, text):
            return super().get_embedding(text) + [0.0] * (self.dimensions - 3)

        def get_dimensions(self):
            return self.dimensions

        def get_provider_info(self):
            return {
                "provider": "deterministic-test",
                "model": "notion-vector-contract",
                "dimensions": self.dimensions,
            }

    embedder = OracleEmbedder()
    vectors = OracleProvider.from_env(
        index_policy="none", in_database_embedding=False, embedding_provider=embedder
    )
    api = NotionAPI()
    client = NotionClient(
        "test-notion-only", session=api, sleep=lambda _: None, clock=lambda: 0
    )
    provider = None
    identifiers = []
    try:
        # Read only the dimension of one existing vector, never its source text.
        with vectors._get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT embedding FROM "
                + vectors._get_table_name(MemoryType.KNOWLEDGE_BASE)
                + " WHERE embedding IS NOT NULL FETCH FIRST 1 ROW ONLY"
            )
            row = cursor.fetchone()
            if row and row[0] is not None:
                embedder.dimensions = len(row[0])
        provider = NotionProvider(
            NotionConfig(
                api.data_source_id,
                token="test-notion-only",
                state_path=tmp_path / "journal.sqlite",
            ),
            vectors,
            client=client,
        )
        for user, content in ((None, "coffee"), ("", "coffee"), ("alice", "tea")):
            identifier = str(uuid.uuid4())
            identifiers.append(identifier)
            assert (
                provider.store(
                    {"id": identifier, "content": content, "user_id": user},
                    MemoryType.KNOWLEDGE_BASE,
                )
                == identifier
            )
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, user_id=None
            )[0]["id"]
            == identifiers[0]
        )
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, user_id=""
            )[0]["id"]
            == identifiers[1]
        )
        assert (
            provider.retrieve_by_query(
                "matcha", MemoryType.KNOWLEDGE_BASE, user_id="alice"
            )[0]["id"]
            == identifiers[2]
        )
        provider.update_by_id(
            identifiers[2], {"content": "coffee"}, MemoryType.KNOWLEDGE_BASE
        )
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, user_id="alice"
            )[0]["content"]
            == "coffee"
        )
        assert provider.delete_by_id(identifiers[2], MemoryType.KNOWLEDGE_BASE)
        assert not provider.retrieve_by_query(
            "espresso", MemoryType.KNOWLEDGE_BASE, user_id="alice"
        )
    finally:
        # Delete only the synthetic vectors in this run's unique Notion namespace.
        if provider is not None:
            for identifier in identifiers:
                vectors.delete_vector(
                    provider._namespace(MemoryType.KNOWLEDGE_BASE), identifier
                )
            provider.close()
        client.close()
        vectors.close()


def test_live_notion_provider(tmp_path):
    if os.getenv("MEMORIZZ_TEST_NOTION_LIVE") != "1":
        pytest.skip(
            "Set MEMORIZZ_TEST_NOTION_LIVE=1 only after authorizing a dedicated Notion test parent page"
        )
    from dotenv import load_dotenv

    from memorizz.memory_provider import (
        FileSystemConfig,
        FileSystemProvider,
        NotionConfig,
        NotionProvider,
    )
    from memorizz.memory_provider.notion import NotionClient, provision_notion_workspace

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    parent = os.getenv("MEMORIZZ_NOTION_TEST_PARENT_PAGE_ID")
    token = os.getenv("NOTION_TOKEN")
    if not parent or not token:
        pytest.fail("NOTION_TOKEN and MEMORIZZ_NOTION_TEST_PARENT_PAGE_ID are required")
    client = NotionClient(token)
    workspace = None
    provider = None
    vectors = FileSystemProvider(
        FileSystemConfig(
            tmp_path / "vectors", embedding_provider=Embedder(), use_faiss=False
        )
    )
    try:
        workspace = provision_notion_workspace(
            parent,
            title="Memorizz disposable integration test " + uuid.uuid4().hex[:8],
            client=client,
        )
        provider = NotionProvider(
            NotionConfig(
                workspace["data_source_id"],
                token=token,
                state_path=tmp_path / "journal.sqlite",
            ),
            vectors,
            client=client,
        )
        identifier = provider.store(
            {
                "content": "coffee",
                "user_id": "notion-integration-test",
                "timestamp": "2026-09-10T10:20:18.488718+01:00",
            },
            MemoryType.KNOWLEDGE_BASE,
        )
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, user_id="notion-integration-test"
            )[0]["id"]
            == identifier
        )
        bounded = provider.query_observability_records(
            MemoryType.KNOWLEDGE_BASE,
            start_time="2026-09-10T09:20:18.488718Z",
            end_time="2026-09-10T09:20:18.488718Z",
        )
        assert [row["id"] for row in bounded["items"]] == [identifier]
        assert not provider.query_observability_records(
            MemoryType.KNOWLEDGE_BASE,
            start_time="2026-09-10T09:20:18.488719Z",
        )["items"]
        provider.update_by_id(identifier, {"content": "tea"}, MemoryType.KNOWLEDGE_BASE)
        assert (
            provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"]
            == "tea"
        )
        assert provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
        assert provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE) == []
    except BaseException as exc:
        workspace = workspace or getattr(exc, "notion_workspace", None)
        raise
    finally:
        try:
            if workspace and workspace.get("root_page_id"):
                # Notion trash is recoverable; never delete the supplied parent page.
                archived = client.request(
                    "PATCH",
                    "/pages/" + workspace["root_page_id"],
                    body={"in_trash": True},
                )
                assert archived.get("in_trash") is True
                print("Archived disposable Notion area: " + workspace["root_page_id"])
        finally:
            if provider is not None:
                provider.close()
            vectors.close()
            client.close()
