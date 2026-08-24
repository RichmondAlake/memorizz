from retry_policy import delays

if delays(3)[:2] != [0.1, 0.2]:
    raise AssertionError("unexpected schedule")
