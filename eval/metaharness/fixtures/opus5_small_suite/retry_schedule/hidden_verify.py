import json
import math
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(workspace))
import retry_policy  # noqa: E402

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


def length_semantics():
    if len(retry_policy.delays(4)) != 3:
        raise AssertionError("four attempts require three retry delays")
    if retry_policy.delays(1) != []:
        raise AssertionError("one attempt requires no retry delay")


def exponential_cap():
    actual = retry_policy.delays(6, base=0.2, cap=0.7)
    if actual != [0.2, 0.4, 0.7, 0.7, 0.7]:
        raise AssertionError(f"unexpected schedule: {actual!r}")


def float_outputs():
    if any(type(value) is not float for value in retry_policy.delays(4, base=1, cap=4)):
        raise AssertionError("all delays must be float")


def base_contract():
    expect(TypeError, lambda: retry_policy.delays(2, base=True))
    expect(TypeError, lambda: retry_policy.delays(2, base="0.1"))
    expect(ValueError, lambda: retry_policy.delays(2, base=0))
    expect(ValueError, lambda: retry_policy.delays(2, base=math.nan))


def cap_contract():
    expect(TypeError, lambda: retry_policy.delays(2, cap=False))
    expect(TypeError, lambda: retry_policy.delays(2, cap=None))
    expect(ValueError, lambda: retry_policy.delays(2, cap=-1))
    expect(ValueError, lambda: retry_policy.delays(2, cap=math.inf))


check("length_semantics", length_semantics)
check("exponential_cap", exponential_cap)
check("float_outputs", float_outputs)
check("attempts_bool", lambda: expect(TypeError, lambda: retry_policy.delays(True)))
check("attempts_type", lambda: expect(TypeError, lambda: retry_policy.delays(2.0)))
check("attempts_value", lambda: expect(ValueError, lambda: retry_policy.delays(0)))
check("base_contract", base_contract)
check("cap_contract", cap_contract)

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
