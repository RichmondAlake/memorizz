import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(workspace))
from cache import Cache  # noqa: E402

results = []


def check(check_id, function):
    try:
        function()
    except Exception as exc:
        results.append((check_id, False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append((check_id, True, None))


def seeded():
    cache = Cache()
    rows = (
        ("u1", "ops", "v1", "u1-ops-v1"),
        ("u2", "ops", "v1", "u2-ops-v1"),
        ("u1", "sales", "v1", "u1-sales-v1"),
        ("u1", "ops", "v2", "u1-ops-v2"),
    )
    for user_id, domain, version, value in rows:
        cache.set(
            "Status",
            value,
            user_id=user_id,
            domain=domain,
            data_version=version,
            ttl=10,
            now=0,
        )
    return cache


def expect_value(user_id, domain, version, expected):
    value = seeded().get(
        " STATUS ",
        user_id=user_id,
        domain=domain,
        data_version=version,
        now=1,
    )
    if value != expected:
        raise AssertionError(f"expected {expected!r}, got {value!r}")


def expiry_boundary():
    cache = Cache()
    cache.set(
        "q",
        "value",
        user_id="u",
        domain="d",
        data_version="v",
        ttl=5,
        now=10,
    )
    if cache.get("q", user_id="u", domain="d", data_version="v", now=15) is not None:
        raise AssertionError("expiry boundary was a hit")
    if len(cache) != 0:
        raise AssertionError("expired entry was not evicted")


def domain_invalidation():
    cache = seeded()
    cache.invalidate(domain="ops")
    if len(cache) != 1:
        raise AssertionError("wrong domain invalidation count")
    expect = cache.get(
        "status",
        user_id="u1",
        domain="sales",
        data_version="v1",
        now=1,
    )
    if expect != "u1-sales-v1":
        raise AssertionError("unmatched domain entry was removed")


def version_invalidation():
    cache = seeded()
    cache.invalidate(data_version="v1")
    if len(cache) != 1:
        raise AssertionError("wrong version invalidation count")
    expect = cache.get(
        "status",
        user_id="u1",
        domain="ops",
        data_version="v2",
        now=1,
    )
    if expect != "u1-ops-v2":
        raise AssertionError("unmatched version entry was removed")


def combined_invalidation():
    cache = seeded()
    cache.invalidate(domain="ops", data_version="v1")
    if len(cache) != 2:
        raise AssertionError("combined filters must use AND semantics")


check("normalized_query", lambda: expect_value("u1", "ops", "v1", "u1-ops-v1"))
check("tenant_isolation", lambda: expect_value("u2", "ops", "v1", "u2-ops-v1"))
check("domain_isolation", lambda: expect_value("u1", "sales", "v1", "u1-sales-v1"))
check("version_isolation", lambda: expect_value("u1", "ops", "v2", "u1-ops-v2"))
check("expiry_boundary", expiry_boundary)
check("domain_invalidation", domain_invalidation)
check("version_invalidation", version_invalidation)
check("combined_invalidation", combined_invalidation)

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
