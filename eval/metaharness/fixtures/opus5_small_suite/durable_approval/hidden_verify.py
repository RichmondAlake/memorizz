import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(workspace))
from approval import (  # noqa: E402
    ApprovalExpired,
    ApprovalStateError,
    ArgumentMismatch,
    Store,
)

results = []


def check(check_id, function):
    try:
        function()
    except Exception as exc:
        results.append((check_id, False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append((check_id, True, None))


def expect(error_type, function):
    try:
        function()
    except error_type:
        return
    except Exception as exc:
        raise AssertionError(f"wrong exception: {type(exc).__name__}") from exc
    raise AssertionError("expected an exception")


def canonical_hash():
    left = Store.hash_arguments({"b": 2, "a": 1})
    right = Store.hash_arguments({"a": 1, "b": 2})
    if left != right:
        raise AssertionError("argument hash is not canonical")


def approver_identity():
    store = Store()
    store.propose("p", {}, now=0, ttl=10)
    expect((TypeError, ValueError), lambda: store.approve("p", approver_id="", now=1))


def approve_expiry():
    store = Store()
    item = store.propose("p", {}, now=0, ttl=10)
    expect(ApprovalExpired, lambda: store.approve("p", approver_id="host", now=10))
    if item.status != "expired":
        raise AssertionError("expired status was not persisted")


def approve_state():
    store = Store()
    store.propose("p", {}, now=0, ttl=10)
    store.approve("p", approver_id="host", now=1)
    expect(
        ApprovalStateError,
        lambda: store.approve("p", approver_id="other", now=2),
    )


def mismatch():
    store = Store()
    store.propose("p", {"x": 1}, now=0, ttl=10)
    store.approve("p", approver_id="host", now=1)
    expect(ArgumentMismatch, lambda: store.consume("p", {"x": 2}, now=2))


def consume_expiry():
    store = Store()
    item = store.propose("p", {}, now=0, ttl=10)
    store.approve("p", approver_id="host", now=1)
    expect(ApprovalExpired, lambda: store.consume("p", {}, now=10))
    if item.status != "expired":
        raise AssertionError("expired status was not persisted")


def single_use():
    store = Store()
    item = store.propose("p", {}, now=0, ttl=10)
    store.approve("p", approver_id="host", now=1)
    store.consume("p", {}, now=2)
    if item.status != "consumed":
        raise AssertionError("consumed status was not persisted")
    expect(ApprovalStateError, lambda: store.consume("p", {}, now=3))


def approver_retained():
    store = Store()
    item = store.propose("p", {}, now=0, ttl=10)
    store.approve("p", approver_id="host-17", now=1)
    store.consume("p", {}, now=2)
    if item.approver_id != "host-17":
        raise AssertionError("approver identity was lost")


check("canonical_hash", canonical_hash)
check("approver_identity", approver_identity)
check("approve_expiry_boundary", approve_expiry)
check("approve_state", approve_state)
check("argument_mismatch", mismatch)
check("consume_expiry_boundary", consume_expiry)
check("single_use", single_use)
check("approver_retained", approver_retained)

print(
    json.dumps(
        {
            "checks": [
                {"id": check_id, "passed": passed, "error": error}
                for check_id, passed, error in results
            ]
        },
        sort_keys=True,
    )
)
