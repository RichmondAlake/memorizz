import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from memorizz.enums import MemoryType
from memorizz.memory_provider import (
    FileSystemConfig,
    FileSystemProvider,
    NotionConfig,
    NotionProvider,
)
from memorizz.memory_provider.notion import (
    NotionAPIError,
    NotionClient,
    NotionIndexingError,
    NotionIntegrityError,
    NotionQueryLimitError,
    NotionWriteUncertain,
)
from tests.fixtures.notion_api import NotionAPI, Response


class Embedder:
    def __init__(self):
        self.calls = []

    def get_embedding(self, text):
        self.calls.append(text)
        text = text.lower()
        return [
            1.0 + sum(text.count(word) for word in ("coffee", "cafe", "espresso")),
            1.0 + sum(text.count(word) for word in ("tea", "matcha", "teapot")),
            0.1,
        ]


@pytest.fixture
def stack(tmp_path):
    api = NotionAPI()
    client = NotionClient(
        "test-notion-token", session=api, sleep=lambda _: None, clock=lambda: 0
    )
    embedder = Embedder()
    vectors = FileSystemProvider(
        FileSystemConfig(
            tmp_path / "vectors", embedding_provider=embedder, use_faiss=False
        )
    )
    config = NotionConfig(
        api.data_source_id,
        token="test-notion-token",
        state_path=tmp_path / "journal.sqlite",
    )
    provider = NotionProvider(config, semantic_provider=vectors, client=client)
    return provider, vectors, api, embedder


def save(provider, content="I prefer coffee", **fields):
    return provider.store({"content": content, **fields}, MemoryType.KNOWLEDGE_BASE)


def page_id(provider, identifier, memory_type=MemoryType.KNOWLEDGE_BASE):
    return provider._state.get(identifier, memory_type.value)["page_id"]


def test_round_trip_and_content_free_vectors(stack):
    provider, vectors, api, embedder = stack
    identifier = save(
        provider,
        user_id="alice",
        memory_id="memory",
        thread_id="thread",
        namespace="personal",
    )
    row = provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert row["content"] == "I prefer coffee"
    hits = provider.retrieve_by_query(
        "espresso",
        MemoryType.KNOWLEDGE_BASE,
        user_id="alice",
        memory_id="memory",
        thread_id="thread",
        namespace="personal",
    )
    assert [hit["id"] for hit in hits] == [identifier]
    assert hits[0]["score"] > 0.9
    assert hits[0]["notion"]["page_id"] == page_id(provider, identifier)
    index_rows = vectors.list_all(MemoryType.KNOWLEDGE_BASE)
    assert len(index_rows) == 1
    assert "I prefer coffee" not in json.dumps(index_rows)
    assert "I prefer coffee" not in provider.config.state_path.read_bytes().decode(
        "utf-8", errors="ignore"
    )
    assert "embedding" not in json.dumps(api.pages)
    assert provider.index_status()["pending"] == 0


@pytest.mark.parametrize("memory_type", list(MemoryType))
def test_every_memory_type_round_trips(stack, memory_type):
    provider, _, _, _ = stack
    data = {
        "content": "coffee",
        "name": "example",
        "user_id": "alice",
        "memory_id": "memory",
    }
    identifier = provider.store(data, memory_type)
    row = provider.retrieve_by_id(identifier, memory_type)
    assert row["content"] == data["content"]
    assert row["name"] == "example"
    assert provider.retrieve_by_name("example", memory_type)["id"] == identifier
    assert len(provider.list_all(memory_type, user_id="alice")) == 1
    assert provider.list_all(memory_type, user_id="bob") == []
    assert provider.delete_by_id(identifier, memory_type)
    assert provider.retrieve_by_id(identifier, memory_type) is None


def test_tenant_thread_namespace_are_filtered_before_top_k(stack):
    provider, _, _, _ = stack
    for i in range(30):
        save(
            provider,
            "espresso espresso",
            user_id="bob",
            memory_id="other",
            thread_id="other",
        )
    target = save(
        provider,
        "coffee",
        user_id="alice",
        memory_id="mine",
        thread_id="thread",
        namespace="ns",
    )
    for kwargs in (
        {"user_id": "alice"},
        {"memory_id": "mine"},
        {"thread_id": "thread"},
        {"namespace": "ns"},
    ):
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, limit=1, **kwargs
            )[0]["id"]
            == target
        )


def test_anonymous_is_not_unscoped(stack):
    provider, _, _, _ = stack
    anonymous = save(provider)
    save(provider, user_id="alice")
    assert [
        row["id"] for row in provider.list_all(MemoryType.KNOWLEDGE_BASE, user_id=None)
    ] == [anonymous]
    assert [
        row["id"]
        for row in provider.retrieve_by_query(
            "espresso", MemoryType.KNOWLEDGE_BASE, user_id=None
        )
    ] == [anonymous]
    assert len(provider.list_all(MemoryType.KNOWLEDGE_BASE)) == 2


def test_updates_replace_embedding_and_deletion_removes_vector(stack):
    provider, vectors, _, _ = stack
    identifier = save(provider)
    assert provider.update_by_id(
        identifier, {"content": "I prefer tea"}, MemoryType.KNOWLEDGE_BASE
    )
    result = provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)
    assert result[0]["content"] == "I prefer tea"
    assert len(vectors.list_all(MemoryType.KNOWLEDGE_BASE)) == 1
    assert provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert vectors.list_all(MemoryType.KNOWLEDGE_BASE) == []
    assert not provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)


def test_human_edits_fail_closed_until_sync(stack):
    provider, _, api, _ = stack
    identifier = save(provider)
    api.edit(page_id(provider, identifier), "content", "I prefer tea")
    assert (
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"]
        == "I prefer tea"
    )
    assert provider.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE) == []
    assert provider.sync_page(page_id(provider, identifier))
    assert (
        provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)[0]["content"]
        == "I prefer tea"
    )


@pytest.mark.parametrize("change", ["delete", "revoke"])
def test_deleted_or_inaccessible_notion_content_never_leaks(stack, change):
    provider, vectors, api, _ = stack
    identifier = save(provider)
    notion_id = page_id(provider, identifier)
    if change == "delete":
        api.pages[notion_id]["in_trash"] = True
    else:
        api.hidden.add(notion_id)
    assert provider.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE) == []
    assert provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE) is None
    provider.sync(MemoryType.KNOWLEDGE_BASE)
    assert not vectors.list_all(MemoryType.KNOWLEDGE_BASE)


def test_index_failure_is_reported_and_repair_survives_restart(stack, monkeypatch):
    provider, vectors, api, _ = stack
    real = vectors.upsert_vector
    monkeypatch.setattr(
        vectors,
        "upsert_vector",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("secret failure detail")),
    )
    with pytest.raises(NotionIndexingError) as caught:
        save(provider)
    identifier = caught.value.record_id
    assert "secret failure" not in str(caught.value)
    assert provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert provider.index_status()["pending"] == 1
    monkeypatch.setattr(vectors, "upsert_vector", real)
    restarted = NotionProvider(provider.config, vectors, client=provider._client)
    assert restarted.repair_index()["repaired"] == 1
    assert (
        restarted.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE)[0]["id"]
        == identifier
    )
    assert len(api.pages) == 1


def test_uncertain_create_is_reconciled_without_duplicate(stack):
    provider, _, api, _ = stack
    identifier = str(uuid.uuid4())

    def applied_timeout(method, path, body):
        if method == "POST" and path == "/pages":
            api.handle(method, path, body, {})
            return requests.Timeout("token must not leak")
        api.failures.append(applied_timeout)

    api.failures.append(applied_timeout)
    with pytest.raises(NotionWriteUncertain):
        save(provider, id=identifier)
    assert len(api.pages) == 1
    assert provider.repair_index()["repaired"] == 1
    assert save(provider, id=identifier) == identifier
    assert len(api.pages) == 1


def test_uncertain_absent_create_does_not_blindly_retry(stack):
    provider, _, api, _ = stack
    identifier = str(uuid.uuid4())
    api.failures.extend(
        [None, requests.Timeout()]
    )  # ID query succeeds, create times out.
    with pytest.raises(NotionWriteUncertain):
        save(provider, id=identifier)
    with pytest.raises(NotionWriteUncertain):
        save(provider, id=identifier)
    assert len(api.pages) == 0
    assert provider.repair_index()["unresolved"] == 1


def test_pagination_and_long_rich_text_are_complete(stack):
    provider, _, api, _ = stack
    api.page_size = 2
    for number in range(5):
        save(provider, "coffee " + str(number))
    content = "coffee ☕ " * 6500
    identifier = save(provider, content)
    assert (
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"]
        == content
    )
    assert len(provider.list_all(MemoryType.KNOWLEDGE_BASE)) == 6
    assert any("/properties/" in path for _, path, _, _ in api.calls)


def test_query_budget_raises_instead_of_claiming_complete(stack):
    provider, _, api, _ = stack
    for i in range(3):
        save(provider, id=str(uuid.uuid4()))
    api.page_size = 1
    provider.config.max_query_pages = 1
    with pytest.raises(NotionQueryLimitError):
        provider.list_all(MemoryType.KNOWLEDGE_BASE)
    with pytest.raises(NotionQueryLimitError):
        provider.delete_all(MemoryType.KNOWLEDGE_BASE)
    assert all(not row["in_trash"] for row in api.pages.values())


def test_scope_tampering_is_not_a_tenant_transfer(stack):
    provider, _, api, _ = stack
    identifier = save(provider, user_id="alice")
    api.edit(page_id(provider, identifier), "user_id", "bob")
    with pytest.raises(NotionIntegrityError):
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)


def test_property_renames_survive_restart(stack):
    provider, vectors, api, _ = stack
    identifier = save(provider)
    api.properties["My readable memory"] = api.properties.pop("Content")
    restarted = NotionProvider(provider.config, vectors, client=provider._client)
    assert (
        restarted.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"]
        == "I prefer coffee"
    )


def test_read_only_blocks_all_mutating_paths(stack):
    provider, _, _, _ = stack
    identifier = save(provider)
    provider.config.read_only = True
    for operation in (
        lambda: save(provider),
        lambda: provider.update_by_id(identifier, {}, MemoryType.KNOWLEDGE_BASE),
        lambda: provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE),
        provider.sync,
        provider.repair_index,
    ):
        with pytest.raises(PermissionError):
            operation()
    assert provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)


def test_no_semantic_provider_is_explicit_not_a_false_miss(stack):
    provider, _, _, _ = stack
    provider.semantic_provider = None
    identifier = save(provider)
    assert provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    with pytest.raises(NotImplementedError, match="semantic_provider"):
        provider.retrieve_by_query("coffee", MemoryType.KNOWLEDGE_BASE)


def test_cas_is_never_emulated_with_notion_read_then_write(stack):
    with pytest.raises(NotImplementedError, match="atomic"):
        stack[0].compare_and_swap_shared_memory("memory", "before", "after")


def test_concurrent_same_id_creates_one_notion_page(stack):
    provider, _, api, _ = stack
    identifier = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: save(provider, id=identifier), range(8)))
    assert results == [identifier] * 8
    assert len(api.pages) == 1


def test_ordered_conversation_history_keeps_tenant_and_thread(stack):
    provider = stack[0]
    for user, thread, timestamp in [
        ("alice", "one", "2026-01-02T00:00:00Z"),
        ("bob", "one", "2026-01-01T00:00:00Z"),
        ("alice", "two", "2026-01-01T00:00:00Z"),
        ("alice", "one", "2026-01-01T00:00:00Z"),
    ]:
        provider.store(
            {
                "content": "coffee",
                "role": "user",
                "user_id": user,
                "thread_id": thread,
                "timestamp": timestamp,
            },
            MemoryType.CONVERSATION_MEMORY,
            memory_id="memory",
        )
    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        "memory", user_id="alice", thread_id="one", limit=1
    )
    assert len(rows) == 1 and rows[0]["timestamp"] == "2026-01-02T00:00:00Z"


def test_transport_retries_rate_limits_not_auth_and_redacts_errors():
    api, sleeps = NotionAPI(), []
    client = NotionClient(
        "notion-secret", session=api, sleep=sleeps.append, clock=lambda: 0
    )
    api.failures = [
        Response(429, {"code": "rate_limited"}, {"Retry-After": "2"}),
        Response(529, {"code": "service_overload"}, {"Retry-After": "3"}),
    ]
    client.request("GET", "/data_sources/" + api.data_source_id)
    assert sleeps == [2.0, 3.0]
    api.failures = [Response(401, {"code": "unauthorized", "message": "notion-secret"})]
    with pytest.raises(NotionAPIError) as caught:
        client.request("GET", "/data_sources/" + api.data_source_id)
    assert "notion-secret" not in str(caught.value)
    assert "notion-secret" not in repr(client)


def test_configuration_does_not_expose_credentials(tmp_path):
    config = NotionConfig(
        str(uuid.uuid4()), token="sensitive-token", state_path=tmp_path / "state"
    )
    assert "sensitive-token" not in repr(config)
    with pytest.raises(ValueError):
        NotionClient("sensitive-token", base_url="https://example.com")


def test_oversized_memory_is_rejected_before_writes(stack):
    provider, _, api, _ = stack
    with pytest.raises(ValueError, match="split"):
        save(provider, "x" * 180_001)
    assert not api.pages


def test_concurrent_partial_updates_do_not_lose_fields(stack):
    provider, _, _, _ = stack
    identifier = save(provider)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(
            executor.map(
                lambda n: provider.update_by_id(
                    identifier, {"field_" + str(n): n}, MemoryType.KNOWLEDGE_BASE
                ),
                range(8),
            )
        )
    row = provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert all(row["field_" + str(n)] == n for n in range(8))


def test_revoked_page_is_never_recreated_by_a_write(stack):
    provider, _, api, _ = stack
    identifier = save(provider)
    api.hidden.add(page_id(provider, identifier))
    with pytest.raises(NotionIntegrityError, match="inaccessible"):
        save(provider, id=identifier)
    assert len(api.pages) == 1


def test_crash_after_remote_create_keeps_a_durable_intent(stack):
    provider, vectors, api, _ = stack
    identifier = str(uuid.uuid4())

    def crash(method, path, body):
        if method == "POST" and path == "/pages":
            api.handle(method, path, body, {})
            raise SystemExit("simulated process crash")
        api.failures.append(crash)

    api.failures.append(crash)
    with pytest.raises(SystemExit):
        save(provider, id=identifier)
    assert len(api.pages) == 1
    restarted = NotionProvider(provider.config, vectors, client=provider._client)
    with pytest.raises(NotionWriteUncertain):
        save(restarted, id=identifier)
    assert restarted.repair_index()["repaired"] == 1
    assert (
        restarted.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE)[0]["id"]
        == identifier
    )


def test_native_observability_pagination_is_bounded_and_scope_bound(stack, monkeypatch):
    provider, _, api, _ = stack
    for index in range(5):
        provider.store(
            {
                "content": "turn",
                "role": "user",
                "user_id": "alice",
                "agent_id": "agent",
                "thread_id": "thread",
                "timestamp": f"2026-01-0{index + 1}T00:00:00Z",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
    monkeypatch.setattr(
        provider,
        "list_all",
        lambda *a, **kw: pytest.fail("Observability must not scan list_all"),
    )
    first = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY, user_id="alice", limit=2
    )
    assert len(first["items"]) == first["scanned_count"] == 2
    assert first["items"][0]["timestamp"].startswith("2026-01-05")
    assert first["next_cursor"]
    second = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY,
        user_id="alice",
        limit=2,
        cursor=first["next_cursor"],
    )
    assert not {row["id"] for row in first["items"]} & {
        row["id"] for row in second["items"]
    }
    with pytest.raises(ValueError, match="scoped"):
        provider.query_observability_records(
            MemoryType.CONVERSATION_MEMORY, user_id="bob", cursor=first["next_cursor"]
        )
    bounded = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY,
        user_id="alice",
        start_time="2026-01-04T00:00:00Z",
    )
    assert len(bounded["items"]) == 2


def test_live_notion_minute_dates_preserve_exact_timestamps_and_integrity(stack):
    provider, _, api, _ = stack
    timestamp = "2026-09-10T10:20:18.488718+01:00"
    identifier = save(provider, timestamp=timestamp)
    page = api.pages[page_id(provider, identifier)]
    assert page["properties"]["p:event_time"]["date"]["start"].endswith(
        "09:20:00.000+00:00"
    )
    assert (
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["timestamp"]
        == timestamp
    )
    assert (
        provider.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE)[0]["id"]
        == identifier
    )
    provider.update_by_id(identifier, {"content": "tea"}, MemoryType.KNOWLEDGE_BASE)
    assert (
        provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)[0]["timestamp"]
        == timestamp
    )
    page["properties"]["p:event_time"]["date"][
        "start"
    ] = "2026-09-10T09:21:00.000+00:00"
    with pytest.raises(NotionIntegrityError, match="event time"):
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)


def test_observability_subminute_bounds_order_and_cursor_scope(stack):
    provider, _, api, _ = stack
    for identifier, timestamp in (
        ("z-before", "2026-09-10T09:20:18.488700Z"),
        ("b-inside", "2026-09-10T10:20:18.488718+01:00"),
        ("a-inside", "2026-09-10T09:20:19.000000Z"),
        ("0-after", "2026-09-10T09:20:19.000001Z"),
    ):
        provider.store(
            {"id": identifier, "content": "turn", "timestamp": timestamp},
            MemoryType.CONVERSATION_MEMORY,
        )
    arguments = {
        "start_time": "2026-09-10T09:20:18.488718Z",
        "end_time": "2026-09-10T09:20:19Z",
    }
    result = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY, **arguments
    )
    assert [row["id"] for row in result["items"]] == ["a-inside", "b-inside"]
    assert result["scanned_count"] == 4
    api.page_size = 1
    first = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY, **arguments
    )
    assert first["items"] == [] and first["next_cursor"]
    with pytest.raises(ValueError, match="scoped"):
        provider.query_observability_records(
            MemoryType.CONVERSATION_MEMORY,
            **{**arguments, "start_time": "2026-09-10T09:20:18.488719Z"},
            cursor=first["next_cursor"],
        )
    second = provider.query_observability_records(
        MemoryType.CONVERSATION_MEMORY, **arguments, cursor=first["next_cursor"]
    )
    assert [row["id"] for row in second["items"]] == ["a-inside"]


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-09-10T09:20:18.488718Z",
        "2026-09-10T10:20:18.488718+01:00",
        "2026-09-10T09:20:18.488718",
        "2026-09-10",
        1789032018.488718,
        1789032018,
    ],
)
def test_precise_timestamp_roundtrip_filter_and_tampering(stack, timestamp):
    provider, _, api, _ = stack
    identifier = save(provider, timestamp=timestamp)
    assert (
        provider.retrieve_by_query({"timestamp": timestamp}, MemoryType.KNOWLEDGE_BASE)[
            0
        ]["timestamp"]
        == timestamp
    )
    api.edit(page_id(provider, identifier), "timestamp", "2026-09-10T09:20:59Z")
    with pytest.raises(NotionIntegrityError, match="timestamp"):
        provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)


def test_webhook_signature_replay_and_latest_content(stack):
    import hashlib
    import hmac

    provider, _, api, embedder = stack
    identifier = save(provider)
    notion_id = page_id(provider, identifier)
    api.edit(notion_id, "content", "tea")
    body = json.dumps(
        {
            "id": str(uuid.uuid4()),
            "type": "page.properties_updated",
            "entity": {"type": "page", "id": notion_id},
        }
    ).encode()
    signature = (
        "sha256=" + hmac.new(b"webhook-secret", body, hashlib.sha256).hexdigest()
    )
    before = len(api.calls)
    with pytest.raises(PermissionError):
        provider.process_webhook(
            body, signature="invalid", verification_token="webhook-secret"
        )
    assert len(api.calls) == before
    assert provider.process_webhook(
        body, signature=signature, verification_token="webhook-secret"
    )["refreshed"]
    calls = len(embedder.calls)
    assert provider.process_webhook(
        body, signature=signature, verification_token="webhook-secret"
    )["duplicate"]
    assert len(embedder.calls) == calls
    assert (
        provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)[0]["content"]
        == "tea"
    )


def test_sync_does_not_reembed_unchanged_memories(stack):
    provider, _, _, embedder = stack
    save(provider)
    calls = len(embedder.calls)
    assert provider.sync(MemoryType.KNOWLEDGE_BASE)["refreshed"] == 0
    assert len(embedder.calls) == calls
    assert provider.sync(MemoryType.KNOWLEDGE_BASE, force=True)["refreshed"] == 1
    assert len(embedder.calls) == calls + 1


def test_switching_semantic_backend_marks_rebuild_and_repairs(stack, tmp_path):
    provider, _, _, _ = stack
    identifier = save(provider)
    replacement = FileSystemProvider(
        FileSystemConfig(
            tmp_path / "replacement-vectors",
            embedding_provider=Embedder(),
            use_faiss=False,
        )
    )
    restarted = NotionProvider(provider.config, replacement, client=provider._client)
    assert restarted.index_status()["pending"] == 1
    with pytest.raises(RuntimeError, match="pending"):
        restarted.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE)
    assert restarted.repair_index()["repaired"] == 1
    assert (
        restarted.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE)[0]["id"]
        == identifier
    )


def test_session_and_application_scopes_do_not_crowd_out_top_k(stack):
    provider, _, _, _ = stack
    save(provider, session_id="other", application_id="other")
    identifier = save(provider, session_id="mine", application_id="app")
    assert (
        provider.retrieve_by_query(
            "espresso",
            MemoryType.KNOWLEDGE_BASE,
            session_id="mine",
            application_id="app",
        )[0]["id"]
        == identifier
    )


def test_immutable_trace_is_first_write_wins(stack):
    provider = stack[0]
    identifier = provider.store(
        {
            "id": "trace",
            "content": "first",
            "immutable_trace": True,
            "record_type": "observability_trace_bundle",
        },
        MemoryType.SHARED_MEMORY,
    )
    provider.store({"id": identifier, "content": "second"}, MemoryType.SHARED_MEMORY)
    assert (
        provider.retrieve_by_id(identifier, MemoryType.SHARED_MEMORY)["content"]
        == "first"
    )
    with pytest.raises(PermissionError):
        provider.update_by_id(
            identifier, {"content": "third"}, MemoryType.SHARED_MEMORY
        )


def test_structured_content_and_long_name_are_not_truncated(stack):
    provider = stack[0]
    identifier = save(provider, {"nested": ["coffee", "tea"]}, name="長い名前" * 200)
    row = provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert row["content"] == {"nested": ["coffee", "tea"]}
    assert row["name"] == "長い名前" * 200
    assert (
        provider.retrieve_by_query("coffee", MemoryType.KNOWLEDGE_BASE)[0]["id"]
        == identifier
    )


def test_namespace_argument_none_means_unscoped(stack):
    provider = stack[0]
    identifier = save(provider, namespace="library")
    assert (
        provider.retrieve_by_query("coffee", MemoryType.KNOWLEDGE_BASE, namespace=None)[
            0
        ]["id"]
        == identifier
    )


def test_arbitrary_global_embeddings_do_not_contaminate_selected_model(stack):
    provider, vectors, _, _ = stack
    save(provider, embedding=[999.0])
    assert len(vectors.list_all(MemoryType.KNOWLEDGE_BASE)[0]["embedding"]) == 3


def test_moved_page_cleans_only_its_vector_reference(stack):
    provider, vectors, api, _ = stack
    identifier = save(provider)
    notion_id = page_id(provider, identifier)
    api.pages[notion_id]["parent"]["data_source_id"] = str(uuid.uuid4())
    assert provider.sync_page(notion_id) is False
    assert not vectors.list_all(MemoryType.KNOWLEDGE_BASE)
    assert api.pages[notion_id]["in_trash"] is False


def test_disable_and_reenable_semantics_requires_rebuild(stack):
    from memorizz.memory_provider.notion import NotionError

    provider, vectors, _, _ = stack
    first = save(provider)
    documents = NotionProvider(provider.config, client=provider._client)
    second = save(documents, "tea")
    documents.repair_index()
    assert documents.index_status()["counts"]["stored"] == 2
    with pytest.raises(NotionError, match="configuration"):
        save(provider)
    restored = NotionProvider(provider.config, vectors, client=provider._client)
    assert restored.index_status()["pending"] == 2
    restored.repair_index()
    assert {
        hit["id"]
        for hit in restored.retrieve_by_query(
            "coffee", MemoryType.KNOWLEDGE_BASE, limit=2
        )
    } == {first, second}


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_crash_during_mutation_has_durable_repair(stack, operation):
    provider, vectors, api, _ = stack
    identifier = save(provider)

    def crash(method, path, body):
        if method == "PATCH":
            api.handle(method, path, body, {})
            raise SystemExit("simulated crash after commit")
        api.failures.append(crash)

    api.failures.append(crash)
    with pytest.raises(SystemExit):
        if operation == "update":
            provider.update_by_id(
                identifier, {"content": "tea"}, MemoryType.KNOWLEDGE_BASE
            )
        else:
            provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    restarted = NotionProvider(provider.config, vectors, client=provider._client)
    assert restarted.index_status()["pending"] == 1
    assert restarted.repair_index()["pending"] == 0
    record = restarted.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert record is None if operation == "delete" else record["content"] == "tea"


def test_old_scan_snapshot_cannot_replace_new_sdk_write(stack):
    provider, _, api, _ = stack
    identifier = save(provider)
    old_page = provider._get_page(page_id(provider, identifier))
    provider.update_by_id(identifier, {"content": "tea"}, MemoryType.KNOWLEDGE_BASE)
    provider._sync_record(old_page)
    assert (
        provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)[0]["content"]
        == "tea"
    )


def test_observability_cap_is_not_reported_as_complete(stack):
    import base64

    provider, _, api, _ = stack
    for _ in range(2):
        provider.store(
            {"content": "trace", "user_id": "alice"}, MemoryType.SHARED_MEMORY
        )
    api.page_size = 1
    page = provider.query_observability_records(
        MemoryType.SHARED_MEMORY, user_id="alice"
    )
    decoded = json.loads(
        base64.urlsafe_b64decode(
            page["next_cursor"] + "=" * (-len(page["next_cursor"]) % 4)
        )
    )
    decoded["scanned"] = 9999
    cursor = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()
    with pytest.raises(NotionQueryLimitError, match="10,000"):
        provider.query_observability_records(
            MemoryType.SHARED_MEMORY, user_id="alice", cursor=cursor
        )


def test_notion_content_aliases_follow_human_edit(stack):
    provider, _, api, _ = stack
    identifier = save(provider, "coffee", text="coffee")
    api.edit(page_id(provider, identifier), "content", "tea")
    row = provider.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
    assert row["content"] == row["text"] == "tea"
    provider.sync_page(page_id(provider, identifier))
    assert (
        provider.retrieve_by_query("tea", MemoryType.KNOWLEDGE_BASE)[0]["text"] == "tea"
    )


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": True},
        {"timeout": "30"},
        {"timeout": float("nan")},
        {"requests_per_second": 4},
        {"max_retries": -1},
        {"max_retries": True},
    ],
)
def test_transport_bounds_are_validated_in_config_even_with_injected_client(
    options, tmp_path
):
    with pytest.raises(ValueError):
        NotionConfig(
            str(uuid.uuid4()), token="test", state_path=tmp_path / "state", **options
        )


@pytest.mark.parametrize("method,payload", [("POST", []), ("POST", {}), ("PATCH", [])])
def test_invalid_successful_write_response_is_uncertain(method, payload):
    api = NotionAPI()
    api.failures.append(Response(200, payload))
    client = NotionClient("test", session=api, max_retries=0)
    with pytest.raises(NotionWriteUncertain):
        client.request(method, "/pages", body={})


def test_abandon_unresolved_create_is_explicit_and_preserves_data(stack):
    provider, _, api, _ = stack
    identifier = "unresolved"
    provider._state.put(identifier, MemoryType.KNOWLEDGE_BASE.value, status="creating")
    with pytest.raises(ValueError, match="confirm"):
        provider.abandon_pending_create(identifier, MemoryType.KNOWLEDGE_BASE)
    provider.abandon_pending_create(identifier, MemoryType.KNOWLEDGE_BASE, confirm=True)
    assert provider.index_status()["pending"] == 0
    with pytest.raises(NotionWriteUncertain):
        save(provider, id=identifier)
    assert not api.pages


def test_skill_vectors_embed_applicability_not_procedure(stack):
    provider, _, _, embedder = stack
    provider.store(
        {
            "name": "Coffee ordering",
            "description": "espresso",
            "preconditions": ["cafe open"],
            "queries": ["order coffee"],
            "content": "mechanism must not be embedded",
        },
        MemoryType.SKILLBOX,
    )
    assert "espresso" in embedder.calls[-1]
    assert "mechanism" not in embedder.calls[-1]
