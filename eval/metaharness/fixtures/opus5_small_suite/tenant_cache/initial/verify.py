from cache import Cache

cache = Cache()
cache.set(
    "Status",
    "ok",
    user_id="u1",
    domain="ops",
    data_version="v1",
    ttl=10,
    now=0,
)
if (
    cache.get(
        " status ",
        user_id="u1",
        domain="ops",
        data_version="v1",
        now=1,
    )
    != "ok"
):
    raise AssertionError("basic cache lookup failed")
