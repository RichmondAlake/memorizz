import ast
import math
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(workspace))
import metrics  # noqa: E402

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
        raise AssertionError(
            f"expected {error_type.__name__}, got {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"expected {error_type.__name__}")


def basic_float():
    value = metrics.ratio(10, 4)
    if value != 2.5 or type(value) is not float:
        raise AssertionError("result must be float 2.5")


def non_finite():
    expect(ValueError, lambda: metrics.ratio(math.nan, 2))
    expect(ValueError, lambda: metrics.ratio(math.inf, 2))


def count_value():
    expect(ValueError, lambda: metrics.ratio(1.0, 0))
    expect(ValueError, lambda: metrics.ratio(1.0, -1))


def optimized_verifier():
    tree = ast.parse((workspace / "verify.py").read_text(encoding="utf-8"))
    if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise AssertionError("verify.py contains a bare assert")


check("basic_float", basic_float)
check("total_bool_type", lambda: expect(TypeError, lambda: metrics.ratio(True, 2)))
check("total_string_type", lambda: expect(TypeError, lambda: metrics.ratio("10", 2)))
check("total_non_finite", non_finite)
check("count_bool_type", lambda: expect(TypeError, lambda: metrics.ratio(1.0, True)))
check("count_float_type", lambda: expect(TypeError, lambda: metrics.ratio(1.0, 2.0)))
check("count_value", count_value)
check("optimized_verifier", optimized_verifier)

import json  # noqa: E402

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
