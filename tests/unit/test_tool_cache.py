"""The MemAgent tool-call cache: what it may reuse, for whom, and for how long."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memorizz.memagent import MemAgent
from memorizz.tool_cache import (
    ToolCache,
    ToolCacheConfig,
    ToolCacheStore,
    callable_fingerprint,
    mcp_tool_is_cacheable,
)
from memorizz.tooling import ToolPolicy, governed_tool

pytestmark = pytest.mark.unit


def _call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def _agent(*tools, cache=True, **kwargs) -> MemAgent:
    agent = MemAgent(tools=list(tools), tool_cache=cache, **kwargs)
    agent.semantic_tool_router.enabled = False
    if agent.tool_cache is not None:
        agent.tool_cache.store = ToolCacheStore()  # isolated from other tests
    return agent


def _run(agent: MemAgent, name: str, arguments: dict, *, user_id=None, call_id="c"):
    messages: list = []
    events: list = []
    # Each call is its own turn: within one turn the tool router already
    # refuses an identical repeat, so the cache matters across turns and runs.
    agent.semantic_tool_router.begin_turn(user_id=user_id)
    agent.set_stream_event_callback(events.append)
    agent._execute_and_record_tool_call(
        _call(name, arguments, call_id), messages, None, user_id
    )
    agent.set_stream_event_callback(None)
    results = [e for e in events if e.get("trace_kind") == "tool_result"]
    cache = next((e.get("cache") for e in results if e.get("cache") is not None), None)
    return messages[-1]["content"], cache


# --- Policy -------------------------------------------------------------


def test_a_cacheable_tool_must_be_safe_to_reuse():
    with pytest.raises(ValueError):
        governed_tool(cacheable=True, side_effects=True)(lambda: None)
    with pytest.raises(ValueError):
        governed_tool(cacheable=True, deterministic=False)(lambda: None)
    with pytest.raises(ValueError):
        ToolPolicy(cacheable=True, requires_approval=True)
    policy = ToolPolicy(cacheable=True, cache_ttl_seconds=60).to_dict()
    assert policy["cacheable"] is True and policy["cache_ttl_seconds"] == 60
    # Saved policies from before this field still load.
    assert (
        ToolPolicy(**{"deterministic": True, "side_effects": False}).cacheable is False
    )


def test_config_values():
    assert ToolCacheConfig.from_value(None) is None
    assert ToolCacheConfig.from_value(False) is None
    assert ToolCacheConfig.from_value({"enabled": False}) is None
    config = ToolCacheConfig.from_value(
        {"ttl_seconds": 30, "scope": "user", "ignored": 1}
    )
    assert config.ttl_seconds == 30 and config.scope == "user"
    assert ToolCacheConfig.from_value(True).to_dict()["mcp"] == "annotated"
    with pytest.raises(ValueError):
        ToolCacheConfig(scope="everyone")
    with pytest.raises(ValueError):
        ToolCacheConfig(mcp="always")


def test_admission_decisions():
    cache = ToolCache(ToolCacheConfig(ttl_seconds=300), store=ToolCacheStore())
    assert cache.decide({"deterministic": True}).reason == "not_cacheable"
    assert cache.decide({"cacheable": True, "side_effects": True}).cacheable is False
    assert (
        cache.decide({"cacheable": True, "requires_approval": True}).cacheable is False
    )
    plain = cache.decide({"cacheable": True, "deterministic": True})
    assert plain.cacheable and plain.ttl_seconds == 300
    assert cache.decide({"cacheable": True, "cache_ttl_seconds": 20}).ttl_seconds == 20
    # A tool's domains cap its freshness.
    assert cache.decide({"cacheable": True, "domains": ["inventory"]}).ttl_seconds == 60
    # MCP: the server's hints decide; "off" never caches.
    assert (
        cache.decide(
            {"deterministic": False, "cacheable": True, "domains": ["mcp"]}, is_mcp=True
        ).ttl_seconds
        == 300
    )
    assert cache.decide({"deterministic": False}, is_mcp=True).cacheable is False
    off = ToolCache(ToolCacheConfig(mcp="off"), store=ToolCacheStore())
    assert off.decide({"cacheable": True}, is_mcp=True).reason == "mcp_caching_off"
    assert mcp_tool_is_cacheable({"readOnlyHint": True, "idempotentHint": True})
    assert not mcp_tool_is_cacheable({"readOnlyHint": True})
    assert mcp_tool_is_cacheable({"read_only_hint": True, "idempotent_hint": True})
    assert not mcp_tool_is_cacheable({"readOnlyHint": True, "idempotentHint": "yes"})


# --- Store ----------------------------------------------------------------


def test_keys_ignore_argument_order_and_separate_users_and_agents():
    store = ToolCacheStore()
    a = ToolCache(ToolCacheConfig(), agent_id="agent-a", store=store)
    b = ToolCache(ToolCacheConfig(), agent_id="agent-b", store=store)
    key = a.key("quote", {"zip": "98101", "kg": 2}, user_id="u1", fingerprint="f")
    assert key == a.key(
        "quote", {"kg": 2, "zip": "98101"}, user_id="u1", fingerprint="f"
    )
    assert key != a.key(
        "quote", {"zip": "98101", "kg": 2}, user_id="u2", fingerprint="f"
    )
    assert key != a.key(
        "quote", {"zip": "98101", "kg": 2}, user_id="u1", fingerprint="g"
    )
    assert key != b.key(
        "quote", {"zip": "98101", "kg": 2}, user_id="u1", fingerprint="f"
    )
    shared_a = ToolCache(ToolCacheConfig(scope="user"), agent_id="agent-a", store=store)
    shared_b = ToolCache(ToolCacheConfig(scope="user"), agent_id="agent-b", store=store)
    assert shared_a.key("q", {}, user_id="u1", fingerprint="f") == shared_b.key(
        "q", {}, user_id="u1", fingerprint="f"
    )


def test_results_expire_are_copied_and_counted():
    now = [1000.0]
    cache = ToolCache(
        ToolCacheConfig(), agent_id="a", store=ToolCacheStore(), clock=lambda: now[0]
    )
    value = {"items": [1, 2]}
    assert cache.store_result(
        "k", value, tool_name="t", user_id=None, ttl_seconds=10, duration_ms=1500
    )
    value["items"].append(3)  # the caller's object, not the stored one
    hit = cache.lookup("k")
    assert hit.result == {"items": [1, 2]} and hit.saved_ms == 1500
    hit.result["items"].clear()  # a hit's copy cannot change the store
    assert cache.lookup("k").result == {"items": [1, 2]}
    now[0] += 11
    assert cache.lookup("k") is None
    stats = cache.statistics()
    assert stats["hits"] == 2 and stats["misses"] == 1 and stats["saved_ms"] == 3000
    assert stats["hit_rate"] == round(2 / 3, 4)


def test_eviction_size_limit_and_invalidation():
    store = ToolCacheStore(max_entries=2)
    cache = ToolCache(ToolCacheConfig(max_result_chars=50), agent_id="a", store=store)
    for key in ("k1", "k2", "k3"):
        cache.store_result(
            key, key, tool_name="t", user_id="u", ttl_seconds=60, duration_ms=1
        )
    assert len(store) == 2 and cache.lookup("k1") is None
    assert (
        cache.store_result(
            "big", "x" * 100, tool_name="t", user_id="u", ttl_seconds=60, duration_ms=1
        )
        is False
    )
    assert cache.statistics()["bypass_reasons"] == {"result_too_large": 1}
    cache.store_result(
        "other", "o", tool_name="other", user_id="v", ttl_seconds=60, duration_ms=1
    )
    # Room for two: k2 (least recently used) made way for "other".
    assert cache.lookup("k2") is None
    assert cache.invalidate("t") == 1
    assert cache.lookup("other").result == "o"
    assert cache.invalidate(user_id="v") == 1 and len(store) == 0


def test_fingerprint_follows_code_and_schema():
    def lookup(order_id: str) -> str:
        return "a"

    first = callable_fingerprint(lookup, {"input_schema": {"x": 1}})

    def lookup(order_id: str) -> str:  # noqa: F811 - same name, new body
        return "b"

    assert callable_fingerprint(lookup, {"input_schema": {"x": 1}}) != first
    assert callable_fingerprint(
        lookup, {"input_schema": {"x": 2}}
    ) != callable_fingerprint(lookup, {"input_schema": {"x": 1}})


# --- Agent ------------------------------------------------------------------


def test_a_repeated_call_runs_once_and_returns_the_same_result():
    runs = []

    @governed_tool(cacheable=True)
    def shipping_quote(destination_zip: str, weight_kg: float) -> dict:
        """Quote standard shipping."""
        runs.append(destination_zip)
        return {"zip": destination_zip, "price_usd": 6.5 + weight_kg}

    agent = _agent(shipping_quote)
    first, info1 = _run(
        agent, "shipping_quote", {"destination_zip": "98101", "weight_kg": 2}
    )
    second, info2 = _run(
        agent,
        "shipping_quote",
        {"weight_kg": 2, "destination_zip": "98101"},
        call_id="c2",
    )
    assert runs == ["98101"]
    assert first == second
    assert info1["status"] == "stored" and info1["ttl_seconds"] == 300
    assert info2["status"] == "hit" and info2["saved_ms"] >= 0
    _run(
        agent,
        "shipping_quote",
        {"destination_zip": "10001", "weight_kg": 2},
        call_id="c3",
    )
    assert runs == ["98101", "10001"]
    stats = agent.tool_cache_stats()
    assert stats["enabled"] and stats["hits"] == 1 and stats["stored"] == 2
    assert agent._last_tool_outcomes[-1]["cache"]["status"] == "stored"


def test_tools_that_did_not_opt_in_or_have_side_effects_always_run():
    runs = []

    def order_status(order_id: str) -> str:
        """Look up an order (no cache opt-in)."""
        runs.append(order_id)
        return "shipped"

    agent = _agent(order_status)
    for i in range(2):
        _, info = _run(agent, "order_status", {"order_id": "1042"}, call_id=f"c{i}")
        assert info is None
    assert runs == ["1042", "1042"]
    assert agent.tool_cache_stats()["bypass_reasons"]["not_cacheable"] == 2


def test_failures_are_not_kept_and_users_do_not_share_results():
    calls = []

    @governed_tool(cacheable=True)
    def inventory(sku: str) -> str:
        """Units in stock."""
        calls.append(sku)
        if len(calls) == 1:
            return "Error: warehouse API timed out"
        return f"{sku}: 12 units"

    agent = _agent(inventory)
    _, info = _run(agent, "inventory", {"sku": "TENT-2"}, user_id="ada")
    assert info == {"status": "miss", "reason": "not_successful"}
    _run(agent, "inventory", {"sku": "TENT-2"}, user_id="ada", call_id="c2")
    _, hit = _run(agent, "inventory", {"sku": "TENT-2"}, user_id="ada", call_id="c3")
    assert hit["status"] == "hit" and len(calls) == 2
    _, other = _run(
        agent, "inventory", {"sku": "TENT-2"}, user_id="grace", call_id="c4"
    )
    assert other["status"] == "stored" and len(calls) == 3
    assert agent.invalidate_tool_cache("inventory", user_id="ada") == 1
    _run(agent, "inventory", {"sku": "TENT-2"}, user_id="ada", call_id="c5")
    assert len(calls) == 4


def test_a_changed_tool_does_not_reuse_old_results():
    @governed_tool(cacheable=True)
    def price(sku: str) -> str:
        """Price."""
        return "old price"

    agent = _agent(price)
    _run(agent, "price", {"sku": "A"})

    @governed_tool(cacheable=True)
    def price(sku: str) -> str:  # noqa: F811 - redeployed with new code
        """Price."""
        return "new price"

    agent.tool_manager.add_tool(price)
    content, info = _run(agent, "price", {"sku": "A"}, call_id="c2")
    assert content == "new price" and info["status"] == "stored"


def test_without_a_tool_cache_nothing_changes():
    runs = []

    @governed_tool(cacheable=True)
    def quote(zip_code: str) -> str:
        """Quote."""
        runs.append(zip_code)
        return "6.50"

    agent = _agent(quote, cache=None)
    assert agent.tool_cache is None
    for i in range(2):
        _, info = _run(agent, "quote", {"zip_code": "1"}, call_id=f"c{i}")
        assert info is None
    assert runs == ["1", "1"] and agent.tool_cache_stats()["enabled"] is False


def test_mcp_tools_are_cacheable_only_when_read_only_and_idempotent():
    agent = _agent()
    annotations = {
        "lookup_order": {"readOnlyHint": True, "idempotentHint": True},
        "search_live": {"readOnlyHint": True},
        "cancel_order": {"readOnlyHint": False, "idempotentHint": True},
    }
    server = SimpleNamespace(name="orders")
    agent.mcp_manager = SimpleNamespace(
        get_server=lambda name: server,
        tool_requires_approval=lambda s, tool: tool == "cancel_order",
        _known_tool=lambda s, tool: {
            "annotations": annotations[tool],
            "inputSchema": {},
        },
        _server_fingerprint=lambda s: "fp",
    )
    agent._mcp_tool_targets = {f"mcp_{t}": ("orders", t) for t in annotations}
    policy = lambda tool: agent._effective_tool_policy(f"mcp_{tool}", {})
    assert policy("lookup_order")["cacheable"] is True
    assert policy("search_live")["cacheable"] is False
    assert policy("cancel_order")["cacheable"] is False
    key, decision = agent._tool_cache_key(
        "mcp_lookup_order", {"id": 1}, policy("lookup_order"), None
    )
    assert key and decision.ttl_seconds == 300  # the MCP domain's freshness
    key, decision = agent._tool_cache_key(
        "mcp_search_live", {}, policy("search_live"), None
    )
    assert key is None and decision.reason == "mcp_not_read_only_and_idempotent"


def test_the_setting_survives_save_and_load(tmp_path):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path, lazy_vector_indexes=True)
    )
    try:
        agent = MemAgent(
            memory_provider=provider, tool_cache={"ttl_seconds": 120, "scope": "user"}
        )
        agent.save()
        loaded = MemAgent.load(agent.agent_id, memory_provider=provider)
        assert loaded.tool_cache_config.ttl_seconds == 120
        assert loaded.tool_cache_config.scope == "user"
        plain = MemAgent(memory_provider=provider)
        plain.save()
        assert (
            MemAgent.load(plain.agent_id, memory_provider=provider).tool_cache is None
        )
    finally:
        provider.close()
