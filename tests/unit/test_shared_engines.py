"""The shared engines behind the de-duplicated helpers, and what the union adds.

``test_dedup_pinned_outputs.py`` proves each wrapper still produces its former
output. This file covers the engines' own API and the deliberate supersets:
every surface now recognises the union of credential names and token shapes,
so a few clearly sensitive keys and values that one surface used to miss are
now caught there too. Nothing non-sensitive changes.
"""

from __future__ import annotations

import logging

import pytest

from memorizz import redaction
from memorizz._registry import ProviderRegistry
from memorizz.channels.whatsapp.numbers import normalize_whatsapp_number
from memorizz.memory_provider.vectors import cosine, flatten_vector
from memorizz.redaction import DROP, KEEP, OMIT, PSEUDONYM, KeyMatcher, RedactionPolicy

# --- Redaction engine -----------------------------------------------------------


@pytest.mark.unit
def test_key_matcher_anchoring_modes():
    exact = KeyMatcher(exact=redaction.SENSITIVE_KEY_NAMES)
    suffix = KeyMatcher(suffix=redaction.SENSITIVE_KEY_NAMES)
    search = KeyMatcher(search=redaction.SENSITIVE_KEY_NAMES)

    assert exact("api_key") and exact(" Token ") and exact("apiKey") and exact("passwd")
    assert not exact("api_token") and not exact("token_count") and not exact("")

    assert suffix("aws_secret_access_key") and suffix("x-api-key") and suffix("token")
    assert not suffix("token_count") and not suffix("sessionToken")
    assert not suffix("my.cookie") and not suffix("")

    assert search("has_secret") and search("tokenizer") and search("credentials_path")
    assert not search("model") and not search("")


@pytest.mark.unit
def test_mask_drop_and_pseudonym_modes():
    keys = KeyMatcher(exact=("password",))
    value = {
        "password": "p",
        "nested": {"password": "q", "name": "n"},
        "items": [{"password": "r"}, "text"],
    }

    assert redaction.redact(value, RedactionPolicy(keys=keys, replacement="x")) == {
        "password": "x",
        "nested": {"password": "x", "name": "n"},
        "items": [{"password": "x"}, "text"],
    }

    dropped = []
    assert redaction.redact(
        value, RedactionPolicy(keys=keys, mode=DROP), on_drop=dropped.append
    ) == {"nested": {"name": "n"}, "items": [{}, "text"]}
    assert dropped == ["password", "nested.password", "items.password"]

    pseudonymised = RedactionPolicy(
        keys=keys, mode=PSEUDONYM, pseudonym=lambda item: f"<{item}>"
    )
    assert redaction.redact(value, pseudonymised)["nested"] == {
        "password": "<q>",
        "name": "n",
    }

    # A sensitive top-level key applies to the whole value.
    assert (
        redaction.redact("p", RedactionPolicy(keys=keys), key="password")
        == "[REDACTED]"
    )
    assert (
        redaction.redact(
            {"a": 1}, RedactionPolicy(keys=keys, mode=DROP), key="password"
        )
        is None
    )
    assert redaction.redact("p", RedactionPolicy(keys=keys), key="name") == "p"
    assert redaction.redact("p", RedactionPolicy(keys=keys), path="root") == "p"


@pytest.mark.unit
def test_preserve_exempt_key_rule_pseudonym_keys_and_values():
    keys = KeyMatcher(search=("token", "secret"))

    preserved = RedactionPolicy(keys=keys, preserve=redaction.is_token_telemetry)
    assert redaction.redact({"input_tokens": 3, "token": "t"}, preserved) == {
        "input_tokens": 3,
        "token": "[REDACTED]",
    }

    # ``exempt`` sees the dotted path of the dict holding the key.
    exempt = RedactionPolicy(
        keys=keys, exempt=lambda parent: parent.split(".")[-1] == "schema"
    )
    assert redaction.redact(
        {
            "schema": {"token": "kept"},
            "tool": {"schema": {"token": "kept"}},
            "args": {"token": "gone"},
        },
        exempt,
    ) == {
        "schema": {"token": "kept"},
        "tool": {"schema": {"token": "kept"}},
        "args": {"token": "[REDACTED]"},
    }

    def rule(key, value, parent):
        if key == "keep_me":
            return KEEP
        if key == "silent" and not parent:
            return OMIT
        return None

    ruled = RedactionPolicy(
        keys=KeyMatcher(exact=("keep_me",)), mode=DROP, key_rule=rule
    )
    dropped = []
    assert redaction.redact(
        {"keep_me": 1, "silent": 2, "inner": {"silent": 3}},
        ruled,
        on_drop=dropped.append,
    ) == {"keep_me": 1, "inner": {"silent": 3}}
    assert dropped == []

    pseudonym_keys = RedactionPolicy(
        keys=keys,
        pseudonym=lambda item: f"<{item}>",
        pseudonym_keys=frozenset({"user_id"}),
        values=((redaction.SENSITIVE_VALUE, "[X]"),),
    )
    assert redaction.redact(
        {
            "user_id": "u",
            "empty": {"user_id": ""},
            "note": "key sk-abcdefghijklmnop ok",
        },
        pseudonym_keys,
    ) == {"user_id": "<u>", "empty": {"user_id": ""}, "note": "key [X] ok"}
    assert redaction.redact("Bearer abcdefghijklmnop", pseudonym_keys) == "[X]"


@pytest.mark.unit
def test_tuple_handling_and_policy_validation():
    keys = KeyMatcher(exact=("secret",))
    value = ("a", {"secret": 1}, [("b",)])
    assert redaction.redact(value, RedactionPolicy(keys=keys)) == [
        "a",
        {"secret": "[REDACTED]"},
        [["b"]],
    ]
    assert redaction.redact(value, RedactionPolicy(keys=keys, tuples="tuple")) == (
        "a",
        {"secret": "[REDACTED]"},
        [("b",)],
    )
    with pytest.raises(ValueError):
        RedactionPolicy(keys=keys, mode="shred")
    with pytest.raises(ValueError):
        RedactionPolicy(keys=keys, tuples="set")
    with pytest.raises(ValueError):
        RedactionPolicy(keys=keys, mode=PSEUDONYM)
    with pytest.raises(ValueError):
        RedactionPolicy(keys=keys, pseudonym_keys=frozenset({"user_id"}))


@pytest.mark.unit
def test_shared_value_patterns():
    assert redaction.SECRET_TOKEN.search("sk-ant-abcdefghijklmnop")
    assert redaction.SECRET_TOKEN.search("pk_live_abcdefghijklmnop")
    assert redaction.SECRET_TOKEN.search("gho_abcdefgh")
    assert not redaction.SECRET_TOKEN.search("task-abcdefghijkl risk_router sk-abcdefg")
    assert redaction.BEARER_TOKEN.search("bearer abcdefgh==")
    assert not redaction.BEARER_TOKEN.search("Bearer short")
    assert redaction.EMAIL.sub("[e]", "mail a.b+c@example.co.uk now") == "mail [e] now"
    assert redaction.EMAIL_ADDRESS.search("x@y.com.z") and not redaction.EMAIL.search(
        "x@y.com.z"
    )
    assert (
        redaction.scrub_text(
            "https://u:p@h/?token=abc",
            ((redaction.URI_CREDENTIALS, "<uri>"), (redaction.QUERY_SECRET, r"\1<q>")),
        )
        == "<uri>h/?token=<q>"
    )


# --- What the union adds to each surface (deliberate supersets) -------------------


@pytest.mark.unit
def test_mcp_redact_now_masks_every_whole_key_credential_name():
    from memorizz.mcp.security import redact

    assert redact(
        {
            "passwd": "p",
            "private_key": "k",
            "apiKey": "a",
            "credentials": "c",
            "secret_access_key": "s",
        }
    ) == {
        "passwd": "***",
        "private_key": "***",
        "apiKey": "***",
        "credentials": "***",
        "secret_access_key": "***",
    }
    # Still whole-key only: compound and telemetry names are untouched.
    assert redact({"api_token": "t", "token_count": 3, "max_tokens": 1}) == {
        "api_token": "t",
        "token_count": 3,
        "max_tokens": 1,
    }


@pytest.mark.unit
def test_metaharness_redact_now_covers_passwd_private_keys_and_vendor_tokens():
    from memorizz.metaharness.security import (
        HarnessSecurityError,
        build_child_environment,
        redact,
    )

    assert redact({"passwd": "p", "private_key": "k", "code_verifier": "v"}) == {
        "passwd": "[REDACTED]",
        "private_key": "[REDACTED]",
        "code_verifier": "[REDACTED]",
    }
    assert redact("push with ghp_abcdefghijklmnop now") == "push with [REDACTED] now"
    assert redact("xoxb-abcdefghijklmnop") == "[REDACTED]"
    assert redact("Bearer abcdefgh==") == "[REDACTED]"
    # The child-environment allowlist uses the same key vocabulary.
    with pytest.raises(HarnessSecurityError):
        build_child_environment(overrides={"PRIVATE_KEY": "x"})
    assert (
        build_child_environment(
            allowed_names=["PRIVATE_KEY"], overrides={"PRIVATE_KEY": "x"}
        )["PRIVATE_KEY"]
        == "x"
    )


@pytest.mark.unit
def test_ui_redact_value_now_masks_bare_token_keys_but_not_token_telemetry():
    from memorizz.ui.security import redact_value

    assert redact_value(
        {
            "token": "t",
            "api_token": "a",
            "bearer_token": "b",
            "code_verifier": "v",
            "credentials": "c",
            "aws-credentials": "d",
            "note": "pk_live_abcdefghijklmnop",
        }
    ) == {
        "token": "[redacted]",
        "api_token": "[redacted]",
        "bearer_token": "[redacted]",
        "code_verifier": "[redacted]",
        "credentials": "[redacted]",
        "aws-credentials": "[redacted]",
        "note": "[redacted-key]",
    }
    telemetry = {
        "token_count": 5,
        "max_tokens": 100,
        "tokens": {"input": 1},
        "tokenizer": "cl100k",
        "credential_id": "id",
        "model": "gpt-4o",
    }
    assert redact_value(telemetry) == telemetry
    # Tuples used to pass through untouched; they are now scrubbed like lists.
    assert redact_value((1, "sk-abcdefghijklmnop")) == [1, "[redacted-key]"]


@pytest.mark.unit
def test_validate_opaque_now_rejects_every_shared_vendor_token_shape():
    from memorizz.observability.privacy import validate_opaque

    for value in (
        "pk_live_abcdefghijklmnop",
        "gho_abcdefgh",
        "xoxp-abcdefgh",
        "tvly-abcdefghijklmnop",
    ):
        with pytest.raises(ValueError):
            validate_opaque(value)
    assert validate_opaque("npm_short") == "npm_short"
    assert validate_opaque("e2b-sandbox-id") == "e2b-sandbox-id"


# --- Cosine similarity --------------------------------------------------------------


@pytest.mark.unit
def test_cosine_mismatch_policies():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    mismatches = [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([0.0, 0.0], [1.0, 1.0]),
        (["a", "b"], [1.0, 2.0]),
        ([[1.0, 2.0]], [1.0, 2.0]),
        ((x for x in [1.0]), [1.0]),
    ]
    for a, b in mismatches:
        assert cosine(a, b) is None
        assert cosine(a, b, on_mismatch="none") is None
        assert cosine(a, b, on_mismatch="zero") == 0.0
        with pytest.raises(ValueError, match="Cosine similarity needs"):
            cosine(a, b, on_mismatch="raise")
    with pytest.raises(ValueError, match="on_mismatch"):
        cosine([1.0], [1.0], on_mismatch="bogus")


@pytest.mark.unit
def test_cosine_handles_large_vectors_and_numpy_arrays():
    np = pytest.importorskip("numpy")
    big = [float(i % 7) + 1.0 for i in range(300)]
    other = [float(i % 5) + 2.0 for i in range(300)]
    expected = float(np.dot(big, other) / (np.linalg.norm(big) * np.linalg.norm(other)))
    assert cosine(big, other) == pytest.approx(expected)
    assert cosine(np.array(big), np.array(other)) == pytest.approx(expected)
    assert cosine(np.array([1.0, 0.0]), [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine(np.array([[1.0, 0.0]]), [1.0, 0.0]) is None
    assert (
        cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]), on_mismatch="zero")
        == 0.0
    )


@pytest.mark.unit
def test_flatten_vector():
    assert list(flatten_vector([[1.0, 2.0], [3.0, 4.0]])) == [1.0, 2.0, 3.0, 4.0]
    assert list(flatten_vector([[1.0, 2.0], [3.0]])) == [1.0, 2.0, 3.0]
    assert list(flatten_vector([])) == []
    assert cosine(flatten_vector([[1.0, 0.0]]), [1.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.unit
def test_semantic_cache_scores_numpy_arrays():
    # ``not vec1`` used to raise on arrays and fall into the 0.0 error path.
    np = pytest.importorskip("numpy")
    from memorizz.short_term_memory.semantic_cache import SemanticCache

    assert SemanticCache._cosine_similarity(
        None, np.array([1.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(1.0)


# --- Provider registry ------------------------------------------------------------


class _Kwargs:
    def __init__(self, *, key=None):
        self.key = key

    def validate_configuration(self):
        return None if self.key else "key required"


class _Config:
    def __init__(self, config=None):
        self.config = dict(config or {})

    def validate_configuration(self):
        return None


@pytest.mark.unit
def test_provider_registry_generic_behaviour(caplog):
    logger = logging.getLogger("memorizz.tests.registry")
    registry = ProviderRegistry(
        logger=logger,
        unknown_message="Unknown thing: %s",
        init_error_message="Bad init '%s': %s",
        normalize=lambda name: name.strip().lower(),
        drop_config_keys=("derived",),
    )
    registry.register(" Kw ", _Kwargs)
    registry.register("cfg", _Config)

    assert registry.get("KW") is _Kwargs and registry.get("") is None
    assert registry.providers == {"kw": _Kwargs, "cfg": _Config}
    assert registry.create("kw", {"key": "k", "derived": True}).key == "k"
    assert registry.create("cfg", {"a": 1, "derived": 1}).config == {"a": 1}

    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert registry.create("nope") is None
    assert caplog.records[-1].getMessage() == "Unknown thing: nope"

    with caplog.at_level(logging.ERROR, logger=logger.name):
        with pytest.raises(TypeError):
            registry.create("kw", {"unexpected": 1})
    assert caplog.records[-1].getMessage() == "Bad init 'kw': ['unexpected']"

    with pytest.raises(ValueError, match="key required"):
        registry.create("kw", {})
    lenient = ProviderRegistry(logger=logger, unknown_message="? %s", validate=False)
    lenient.register("kw", _Kwargs)
    assert lenient.create("kw", {}).key is None


# --- WhatsApp numbers ---------------------------------------------------------------


@pytest.mark.unit
def test_normalize_whatsapp_number_strict_and_lenient():
    assert normalize_whatsapp_number("whatsapp:+1 (555) 123-4567") == "+15551234567"
    assert normalize_whatsapp_number(" 15551234567 ") == "+15551234567"
    with pytest.raises(ValueError, match="'from' is required"):
        normalize_whatsapp_number("", field_name="from")
    with pytest.raises(ValueError, match="to must be an E.164 number"):
        normalize_whatsapp_number("abc", field_name="to")
    assert normalize_whatsapp_number("abc", strict=False) == "+abc"
    assert normalize_whatsapp_number("", strict=False) == "+"
    assert normalize_whatsapp_number("++1", strict=False) == "++1"
    assert normalize_whatsapp_number("whatsapp:+1", strict=False) == "+1"


@pytest.mark.unit
def test_message_handler_rejects_non_e164_senders(monkeypatch):
    # The handler used to key memories by "whatsapp_+<anything>"; it is now
    # strict like the Twilio sender and the automation runner, and the inbound
    # pipeline reports the failure instead of running the agent.
    from memorizz.channels.whatsapp import message_handler

    with pytest.raises(ValueError, match="from must be an E.164 number"):
        message_handler.normalize_phone_to_memory_id("abc")
    assert (
        message_handler.normalize_phone_to_memory_id("whatsapp:+1 555")
        == "whatsapp_+1555"
    )

    class _Agent:
        def run(self, *_args, **_kwargs):
            raise AssertionError("the agent must not run for an invalid sender")

    monkeypatch.setattr(message_handler.MemAgent, "load", lambda *_a, **_k: _Agent())
    result = message_handler.process_incoming_message(
        {"from": "not-a-number", "body": "hi", "message_sid": "SM1"}, "agent", object()
    )
    assert result["success"] is False
    assert "from must be an E.164 number" in result["error"]
