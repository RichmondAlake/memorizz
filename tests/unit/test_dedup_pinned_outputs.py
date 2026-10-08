"""Pinned outputs of the de-duplicated redaction, cosine, registry and WhatsApp helpers.

Every expected value below was captured from the original per-module
implementations before they were folded into shared engines
(``memorizz.redaction``, ``memorizz.memory_provider.vectors.cosine``,
``memorizz._registry`` and ``memorizz.channels.whatsapp.numbers``). The thin
wrappers must keep producing exactly these results.
"""

from __future__ import annotations

import logging

import pytest

# --- Redaction ----------------------------------------------------------------

MCP_CORPUS = [
    (
        {
            "api_key": "sk-abc",
            "Authorization": "Bearer x",
            " token ": "t",
            "name": "search",
            "nested": {
                "client_secret": "s",
                "items": [{"password": "p", "count": 3}],
                "pair": ("cookie", {"cookie": "c"}),
            },
        },
        {
            "api_key": "***",
            "Authorization": "***",
            " token ": "***",
            "name": "search",
            "nested": {
                "client_secret": "***",
                "items": [{"password": "***", "count": 3}],
                "pair": ("cookie", {"cookie": "***"}),
            },
        },
    ),
    ([{"refresh_token": "r"}, "plain", 1], [{"refresh_token": "***"}, "plain", 1]),
    (
        ("access_token", {"access_token": "a"}),
        ("access_token", {"access_token": "***"}),
    ),
    (
        {
            "token_count": 5,
            "api_token": "keep-as-is",
            "secrets": ["x"],
            "code_verifier": "v",
            "bearer_token": "b",
            "secret": {"deep": "s"},
            "max_tokens": 100,
            "TOKEN": "upper",
        },
        {
            "token_count": 5,
            "api_token": "keep-as-is",
            "secrets": ["x"],
            "code_verifier": "***",
            "bearer_token": "***",
            "secret": "***",
            "max_tokens": 100,
            "TOKEN": "***",
        },
    ),
    ("sk-abcdefghijklmnop", "sk-abcdefghijklmnop"),
    (None, None),
    (
        {"url": "https://user:pw@host/?api_key=abc", "mail": "a@b.com", 7: "int-key"},
        {"url": "https://user:pw@host/?api_key=abc", "mail": "a@b.com", "7": "int-key"},
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", MCP_CORPUS)
def test_mcp_redact_pinned(value, expected):
    from memorizz.mcp.security import redact

    result = redact(value)
    assert result == expected
    assert type(result) is type(expected)


META_CORPUS = [
    (
        {
            "input_tokens": 120,
            "input_tokens_mean": 118,
            "output_tokens_sample_stdev": 2.5,
            "total_tokens_mean_delta": -320,
            "input_tokens_paired_deltas": [-300, -340],
            "output_tokens": "8",
            "input_tokens_details": {"cached_tokens": 40},
            "cache_write_input_tokens": 12,
            "reasoning_output_tokens": 4,
            "candidate_tokens": 72,
            "context_tokens": 48,
            "tokens_used": 40,
            "tokens_saved": 32,
            "token_reporting": True,
            "max_output_tokens": 1_000,
            "evidence_token_budget": 900,
            "per_million_tokens": {"input": 0.2, "output": 1.2},
            "token_estimate": 24,
            "token_estimates": [24, 36],
            "api_token": "opaque-credential-value",
            "token": "another-opaque-credential",
        },
        {
            "input_tokens": 120,
            "input_tokens_mean": 118,
            "output_tokens_sample_stdev": 2.5,
            "total_tokens_mean_delta": -320,
            "input_tokens_paired_deltas": [-300, -340],
            "output_tokens": "8",
            "input_tokens_details": {"cached_tokens": 40},
            "cache_write_input_tokens": 12,
            "reasoning_output_tokens": 4,
            "candidate_tokens": 72,
            "context_tokens": 48,
            "tokens_used": 40,
            "tokens_saved": 32,
            "token_reporting": True,
            "max_output_tokens": 1000,
            "evidence_token_budget": 900,
            "per_million_tokens": {"input": 0.2, "output": 1.2},
            "token_estimate": 24,
            "token_estimates": [24, 36],
            "api_token": "[REDACTED]",
            "token": "[REDACTED]",
        },
    ),
    (
        {
            "totalTokens": 12,
            "inputTokens": 3,
            "accessToken": "abc",
            "sessionToken": "1234",
        },
        {
            "totalTokens": 12,
            "inputTokens": 3,
            "accessToken": "[REDACTED]",
            "sessionToken": "[REDACTED]",
        },
    ),
    (
        {
            "env": {"OPENAI_API_KEY": "sk-live-abcdefghijklmnop", "HOME": "/home/u"},
            "stdout": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz done",
            "tokens": {"input": 1, "output": 2},
            "token_usage": {"input": "x"},
            "tuple": (1, "pk_live_abcdefghijklmnop"),
            "cookie_jar": "c",
            "credentials_path": "/p",
            "has_secret": False,
            "max_tokens": 100,
            "tokenizer": "cl100k",
            "model": "opus",
            "note": "e2b_abcdefghijklmnop npm_abcdefghijklmnop pypi-abcdefghijklmnop",
        },
        {
            "env": {"OPENAI_API_KEY": "[REDACTED]", "HOME": "/home/u"},
            "stdout": "Authorization: [REDACTED] done",
            "tokens": {"input": 1, "output": 2},
            "token_usage": "[REDACTED]",
            "tuple": [1, "[REDACTED]"],
            "cookie_jar": "[REDACTED]",
            "credentials_path": "[REDACTED]",
            "has_secret": "[REDACTED]",
            "max_tokens": "[REDACTED]",
            "tokenizer": "[REDACTED]",
            "model": "opus",
            "note": "[REDACTED] [REDACTED] [REDACTED]",
        },
    ),
    ("credential=sk-test-abcdefghijklmnop", "credential=[REDACTED]"),
    ("opus5_task_native_small_suite_v1", "opus5_task_native_small_suite_v1"),
    ("risk_router", "risk_router"),
    ("task-abcdefghijkl", "task-abcdefghijkl"),
    ("mongodb://operator:password@host/db", "mongodb://operator:password@host/db"),
    ("a@b.com", "a@b.com"),
    ("Bearer short", "Bearer short"),
    (12, 12),
    (None, None),
    ([{"Secret": 1}, ("x",)], [{"Secret": "[REDACTED]"}, ["x"]]),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", META_CORPUS)
def test_metaharness_redact_pinned(value, expected):
    from memorizz.metaharness.security import redact

    result = redact(value)
    assert result == expected
    assert type(result) is type(expected)


ARCHIVE_CORPUS = [
    (
        {
            "_id": "r1",
            "model": "gpt",
            "api_key": "k",
            "llm_config": {"model": "gpt", "api_key": "k", "x_api_key": "k2"},
            "embedding": [1.0, 0.0],
            "vector": [0.5],
            "content": {"vector": [1], "embedding": [2], "passwd": "p"},
            "parameters": {
                "properties": {"api_key": {"type": "string"}, "token": {}},
                "$defs": {"secret": {"type": "string"}},
                "required": ["api_key"],
            },
            "items": [{"token": "t", "Cookie": "c", "credential": "x"}],
            "aws_secret_access_key": "s",
            "private-key": "pk",
            "tokens": 5,
            "token_count": 3,
            "access_token": "a",
            "my.cookie": "c",
            "secret_name": "name",
            "accessToken": "camel",
            "sessionToken": "camel2",
            "AUTHORIZATION": "A",
        },
        False,
        {
            "_id": "r1",
            "llm_config": {"model": "gpt"},
            "content": {},
            "parameters": {
                "properties": {"api_key": {"type": "string"}, "token": {}},
                "$defs": {"secret": {"type": "string"}},
                "required": ["api_key"],
            },
            "items": [{}],
            "tokens": 5,
            "token_count": 3,
            "my.cookie": "c",
            "secret_name": "name",
            "sessionToken": "camel2",
        },
        [
            "model",
            "api_key",
            "llm_config.api_key",
            "llm_config.x_api_key",
            "content.passwd",
            "items.token",
            "items.Cookie",
            "items.credential",
            "aws_secret_access_key",
            "private-key",
            "access_token",
            "accessToken",
            "AUTHORIZATION",
        ],
    ),
    (
        {"embedding": [1.0], "nested": {"vector": [2.0], "model": "m"}},
        True,
        {"embedding": [1.0], "nested": {"vector": [2.0], "model": "m"}},
        [],
    ),
    ([{"password": "x"}, "text", 3], False, [{}, "text", 3], ["password"]),
    ("plain", False, "plain", []),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,embeddings,expected,expected_removals", ARCHIVE_CORPUS)
def test_memory_archive_sanitize_pinned(value, embeddings, expected, expected_removals):
    from memorizz.memory_archive import _sanitize

    removals = []
    assert _sanitize(value, removals, embeddings=embeddings) == expected
    assert removals == expected_removals


@pytest.fixture
def pinned_pseudonyms(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_PSEUDONYM_KEY", "pinned-key")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_SCOPE", "pinned")


UI_CONTENT = (
    "Authorization: Bearer abcdefghijklmnop; key sk-abcdefghijklmnop; "
    "gh ghp_abcdefghijklmnop; slack xoxb-abcdefghijklmnop; "
    "mongodb://operator:password@host/db?api_key=zzz&x=1 mail a.b+c@example.co.uk end"
)
UI_CONTENT_REDACTED = (
    "Authorization: Bearer [redacted]; key [redacted-key]; gh [redacted-key]; "
    "slack [redacted-key]; mongodb://[redacted]@host/db?api_key=[redacted]&x=1 "
    "mail [redacted-email] end"
)
UI_CORPUS = [
    (
        {
            "api_key": "k",
            "Authorization": {"nested": "x"},
            "password": ["p"],
            "passwd": "p",
            "secret_name": "name",
            "cookie": "c",
            "private_key": "pk",
            "access_token": "a",
            "refreshToken": "r",
            "user_id": "alice",
            "userId": "bob",
            "token_count": 5,
            "max_tokens": 100,
            "tokens": {"input": 1},
            "tokenizer": "cl100k",
            "user_id_hash": "h",
            "content": UI_CONTENT,
            "empty_user": {"user_id": None, "userId": ""},
            "list": [{"user_id": "carol"}, "sk-ant-abcdefghijklmnop", 3, None],
            "task": "task-abcdefghijkl risk_router",
            "model": "gpt-4o",
            "oracle_password": "p",
        },
        {
            "api_key": "[redacted]",
            "Authorization": "[redacted]",
            "password": "[redacted]",
            "passwd": "[redacted]",
            "secret_name": "[redacted]",
            "cookie": "[redacted]",
            "private_key": "[redacted]",
            "access_token": "[redacted]",
            "refreshToken": "[redacted]",
            "user_id": "user:f845508afa83f2b2",
            "userId": "user:804da59fddf268e9",
            "token_count": 5,
            "max_tokens": 100,
            "tokens": {"input": 1},
            "tokenizer": "cl100k",
            "user_id_hash": "h",
            "content": UI_CONTENT_REDACTED,
            "empty_user": {"user_id": None, "userId": ""},
            "list": [{"user_id": "user:dacbd7dd124ce601"}, "[redacted-key]", 3, None],
            "task": "task-abcdefghijkl risk_router",
            "model": "gpt-4o",
            "oracle_password": "[redacted]",
        },
    ),
    ("plain text", "plain text"),
    ("sk-abcdefghijklmnop", "[redacted-key]"),
    (None, None),
    (42, 42),
    ([1, "ghp_abcdefghijklmnop"], [1, "[redacted-key]"]),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", UI_CORPUS)
def test_ui_redact_value_pinned(pinned_pseudonyms, value, expected):
    from memorizz.ui.security import redact_value

    assert redact_value(value) == expected


@pytest.mark.unit
def test_ui_redact_value_with_key_and_scalars_pinned(pinned_pseudonyms):
    from memorizz.ui.security import _redact_scalar, redact_value

    assert redact_value("x", key="password") == "[redacted]"
    assert redact_value({"a": 1}, key="api_key") == "[redacted]"
    assert redact_value("alice", key="user_id") == "user:f845508afa83f2b2"
    assert redact_value("", key="user_id") == ""
    assert redact_value("y", key="name") == "y"
    assert (
        _redact_scalar("Bearer abcdefgh and Bearer short")
        == "Bearer [redacted] and Bearer short"
    )
    assert _redact_scalar(
        "https://u:p@h/?token=abc#frag&secret=1 ftp://x:y@z x@y.org a@b.c"
    ) == (
        "https://[redacted]@h/?token=[redacted]#frag&secret=[redacted] "
        "ftp://[redacted]@z [redacted-email] a@b.c"
    )
    assert _redact_scalar(42) == 42


OPAQUE_CORPUS = [
    ("agent-123", True),
    ("thread_abc", True),
    ("user:42", True),
    ("sk_router", True),
    ("task-abcdefghijkl", True),
    ("risk_router", True),
    ("sk-abcdefg", True),
    ("plain words here", True),
    (12, True),
    (None, True),
    ("a@b.com", False),
    ("Bearer x", False),
    ("bearer tok", False),
    ("sk-abcdefgh", False),
    ("ghp_abcdefgh", False),
    ("xoxb-abcdefgh", False),
    ("https://x", False),
    ("file://x", False),
    ("ab://", False),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,accepted", OPAQUE_CORPUS)
def test_validate_opaque_pinned(value, accepted):
    from memorizz.observability.privacy import validate_opaque

    if accepted:
        assert validate_opaque(value) == value
    else:
        with pytest.raises(ValueError):
            validate_opaque(value)


# --- Cosine similarity --------------------------------------------------------

BIG_A = [((i * 7) % 11) / 10.0 + 0.1 for i in range(100)]
BIG_B = [((i * 3) % 13) / 10.0 + 0.05 for i in range(100)]
COSINE_INPUTS = [
    ([1.0, 0.0], [1.0, 0.0]),
    ([1.0, 0.0], [0.0, 1.0]),
    ([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]),
    ([0.6, 0.8], [0.8, 0.6]),
    ([1, 2, 3], [4, 5, 6]),
    ([1.0, 0.0], [1.0, 0.0, 0.0]),
    ([], [1.0]),
    ([1.0], []),
    ([0.0, 0.0], [1.0, 1.0]),
    ([1.0, 1.0], [0.0, 0.0]),
    (BIG_A, BIG_B),
    (BIG_A, BIG_A),
    ([[1.0, 0.0]], [1.0, 0.0]),
    ([1.0, 0.0], [[0.0, 1.0]]),
]
# Sites that return 0.0 on any mismatch.
COSINE_ZERO = [
    1.0,
    0.0,
    -1.0,
    0.96,
    0.9746318461970762,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7714685693782471,
    1.0,
    0.0,
    0.0,
]
# Sites that return None on any mismatch; the filesystem provider additionally
# flattens nested [[...]] embeddings before comparing them.
COSINE_NONE = [
    1.0,
    0.0,
    -1.0,
    0.96,
    0.9746318461970762,
    None,
    None,
    None,
    None,
    None,
    0.7714685693782471,
    1.0,
    None,
    None,
]
COSINE_FILESYSTEM = [
    1.0,
    0.0,
    -1.0,
    0.96,
    0.9746318461970762,
    None,
    None,
    None,
    None,
    None,
    0.7714685693782471,
    1.0,
    1.0,
    0.0,
]


def _assert_scores(results, expected, *, tolerance):
    assert len(results) == len(expected)
    for result, wanted in zip(results, expected):
        if wanted is None:
            assert result is None
        else:
            assert result == pytest.approx(wanted, rel=tolerance, abs=tolerance)


@pytest.mark.unit
def test_semantic_cache_cosine_pinned():
    from memorizz.short_term_memory.semantic_cache import SemanticCache

    results = [SemanticCache._cosine_similarity(None, a, b) for a, b in COSINE_INPUTS]
    _assert_scores(results, COSINE_ZERO, tolerance=1e-9)
    assert all(isinstance(value, float) for value in results)


@pytest.mark.unit
def test_skillbox_cosine_pinned():
    from memorizz.long_term.procedural.skillbox.skillbox import _cosine

    _assert_scores(
        [_cosine(a, b) for a, b in COSINE_INPUTS], COSINE_ZERO, tolerance=1e-9
    )


@pytest.mark.unit
def test_context_dedup_cosine_pinned():
    from memorizz.memagent.utils.context_dedup import cosine_similarity

    results = [cosine_similarity(a, b) for a, b in COSINE_INPUTS]
    _assert_scores(results, COSINE_NONE, tolerance=1e-9)


@pytest.mark.unit
def test_filesystem_cosine_pinned():
    from memorizz.memory_provider.filesystem.provider import FileSystemProvider

    results = [
        FileSystemProvider._cosine_similarity(None, a, b) for a, b in COSINE_INPUTS
    ]
    # The original computed in float32; pin at float32 precision.
    _assert_scores(results, COSINE_FILESYSTEM, tolerance=1e-6)


_VOCAB = {}


def _fake_embed(sentence):
    words = sentence.lower().split()
    vec = [0.0] * 8
    for word in words:
        idx = _VOCAB.setdefault(word.strip(".!?"), len(_VOCAB)) % 8
        vec[idx] += 1.0
    return vec


SEMANTIC_CORPUS = [
    (
        "Cats purr. Cats sleep. Dogs bark! Dogs run? Birds fly.",
        95.0,
        ["Cats purr. Cats sleep.", "Dogs bark! Dogs run?", "Birds fly."],
    ),
    (
        "Cats purr. Cats sleep. Dogs bark! Dogs run? Birds fly.",
        50.0,
        ["Cats purr. Cats sleep.", "Dogs bark! Dogs run?", "Birds fly."],
    ),
    (
        "Cats purr. Cats sleep. Dogs bark! Dogs run? Birds fly.",
        0.0,
        ["Cats purr.", "Cats sleep.", "Dogs bark!", "Dogs run?", "Birds fly."],
    ),
    ("Single sentence only.", 95.0, ["Single sentence only."]),
    ("", 95.0, []),
    (
        "Alpha beta. Gamma delta. Alpha beta. Zeta eta.",
        75.0,
        ["Alpha beta.", "Gamma delta.", "Alpha beta.", "Zeta eta."],
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("corpus,percentile,expected", SEMANTIC_CORPUS)
def test_knowledge_base_semantic_chunking_pinned(corpus, percentile, expected):
    from memorizz.long_term.semantic.knowledge_base import _chunk_semantic

    _VOCAB.clear()
    assert (
        _chunk_semantic(corpus, percentile, embedding_function=_fake_embed) == expected
    )


# --- Provider registries ------------------------------------------------------


class _KwargsProvider:
    provider_name = "pinned_kwargs"

    def __init__(self, *, api_key=None, region="eu"):
        self.api_key = api_key
        self.region = region

    def validate_configuration(self):
        return "api_key missing" if not self.api_key else None


class _ConfigOnlyProvider:
    provider_name = "pinned_config"

    def __init__(self, config=None):
        self.config = dict(config or {})

    def validate_configuration(self):
        return None


class _NoArgsProvider:
    provider_name = "pinned_noargs"

    def __init__(self):
        pass

    def validate_configuration(self):
        return None


def _registry_cases():
    from memorizz.browser_control import base as browser
    from memorizz.internet_access import base as internet
    from memorizz.sandbox import base as sandbox

    return [
        pytest.param(
            sandbox,
            sandbox.create_sandbox_provider,
            "Unknown sandbox provider: %s",
            "Failed to initialize sandbox provider '%s' with config keys: %s",
            True,
            id="sandbox",
        ),
        pytest.param(
            internet,
            internet.create_internet_access_provider,
            "Unknown internet access provider: %s",
            "Failed to initialize provider '%s' with config keys: %s",
            False,
            id="internet",
        ),
        pytest.param(
            browser,
            browser.create_browser_control_provider,
            "Unknown browser-control provider: %s",
            None,
            True,
            id="browser",
        ),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "module,create,unknown_message,failure_message,validates", _registry_cases()
)
def test_provider_registries_pinned(
    caplog, module, create, unknown_message, failure_message, validates
):
    registry = module._PROVIDER_REGISTRY
    before = dict(registry)
    try:
        module.register_provider("Pinned-Kwargs", _KwargsProvider)
        module.register_provider("pinned_config", _ConfigOnlyProvider)
        module.register_provider("pinned_noargs", _NoArgsProvider)

        assert module.get_provider_class("pinned-kwargs") is _KwargsProvider
        assert module.get_provider_class("PINNED-KWARGS") is _KwargsProvider
        assert module.get_provider_class("") is None
        assert module.get_provider_class("missing") is None
        if module.__name__.endswith("browser_control.base"):
            # The browser registry also ignores separators and whitespace.
            assert module.get_provider_class(" pinned_kwargs ") is _KwargsProvider
            assert module.get_provider_class("pinnedkwargs") is _KwargsProvider
        else:
            assert module.get_provider_class("pinnedkwargs") is None

        provider = create("pinned-kwargs", {"api_key": "k", "region": "us"})
        assert isinstance(provider, _KwargsProvider)
        assert (provider.api_key, provider.region) == ("k", "us")

        provider = create("pinned_config", {"anything": 1})
        assert isinstance(provider, _ConfigOnlyProvider)
        assert provider.config == {"anything": 1}

        assert isinstance(create("pinned_noargs"), _NoArgsProvider)
        assert isinstance(create("pinned_noargs", None), _NoArgsProvider)

        with caplog.at_level(logging.WARNING, logger=module.__name__):
            assert create("missing", {"x": 1}) is None
        assert [(r.name, r.getMessage()) for r in caplog.records] == [
            (module.__name__, unknown_message % "missing")
        ]
        caplog.clear()

        with caplog.at_level(logging.ERROR, logger=module.__name__):
            with pytest.raises(TypeError):
                create("pinned_noargs", {"unexpected": 1})
        expected_logs = (
            [(module.__name__, failure_message % ("pinned_noargs", ["unexpected"]))]
            if failure_message
            else []
        )
        assert [(r.name, r.getMessage()) for r in caplog.records] == expected_logs

        if validates:
            with pytest.raises(ValueError, match="api_key missing"):
                create("pinned-kwargs", {})
        else:
            assert isinstance(create("pinned-kwargs", {}), _KwargsProvider)
    finally:
        registry.clear()
        registry.update(before)


@pytest.mark.unit
def test_internet_registry_drops_derived_config_keys_pinned():
    from memorizz.internet_access import base as internet

    registry = internet._PROVIDER_REGISTRY
    before = dict(registry)
    try:
        internet.register_provider("pinned_kwargs", _KwargsProvider)
        provider = internet.create_internet_access_provider(
            "pinned_kwargs", {"api_key": "k", "api_key_set": True}
        )
        assert provider.api_key == "k"
    finally:
        registry.clear()
        registry.update(before)


# --- WhatsApp numbers -----------------------------------------------------------

E164_MESSAGE = (
    "to must be an E.164 number (e.g. +15551234567) optionally prefixed with "
    "'whatsapp:'. Got: {raw!r}"
)
WHATSAPP_VALID = [
    ("+15551234567", "+15551234567"),
    ("whatsapp:+1 (555) 123-4567", "+15551234567"),
    ("15551234567", "+15551234567"),
    ("WhatsApp: 1555 ", "+1555"),
    ("  +44 20-7946 0958 ", "+442079460958"),
    ("WHATSAPP:+1", "+1"),
]
WHATSAPP_INVALID = [
    ("abc", E164_MESSAGE.format(raw="abc")),
    ("+1-555-abc", E164_MESSAGE.format(raw="+1-555-abc")),
    ("whatsapp:", E164_MESSAGE.format(raw="whatsapp:")),
    ("+", E164_MESSAGE.format(raw="+")),
    ("++15551234567", E164_MESSAGE.format(raw="++15551234567")),
    ("whatsapp:whatsapp:+1", E164_MESSAGE.format(raw="whatsapp:whatsapp:+1")),
]


@pytest.mark.unit
@pytest.mark.parametrize("value,number", WHATSAPP_VALID)
def test_whatsapp_valid_numbers_pinned(value, number):
    from memorizz.automation.runner import _normalize_whatsapp_recipient
    from memorizz.channels.whatsapp.message_handler import normalize_phone_to_memory_id
    from memorizz.channels.whatsapp.twilio import _normalize_whatsapp_address

    assert _normalize_whatsapp_address(value, field_name="to") == f"whatsapp:{number}"
    assert _normalize_whatsapp_recipient(value) == f"whatsapp:{number}"
    assert normalize_phone_to_memory_id(value) == f"whatsapp_{number}"


@pytest.mark.unit
@pytest.mark.parametrize("value,message", WHATSAPP_INVALID)
def test_whatsapp_invalid_numbers_pinned(value, message):
    from memorizz.automation.runner import _normalize_whatsapp_recipient
    from memorizz.channels.whatsapp.twilio import _normalize_whatsapp_address

    with pytest.raises(ValueError) as excinfo:
        _normalize_whatsapp_address(value, field_name="to")
    assert str(excinfo.value) == message
    assert _normalize_whatsapp_recipient(value) == ""


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", None])
def test_whatsapp_empty_numbers_pinned(value):
    from memorizz.automation.runner import _normalize_whatsapp_recipient
    from memorizz.channels.whatsapp.twilio import _normalize_whatsapp_address

    with pytest.raises(ValueError) as excinfo:
        _normalize_whatsapp_address(value, field_name="to")
    assert str(excinfo.value) == "'to' is required"
    assert _normalize_whatsapp_recipient(value) == ""
