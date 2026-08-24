from metrics import ratio

assert ratio(10.0, 2) == 5.0
try:
    ratio(1.0, 0)
except ValueError:
    pass
else:
    raise AssertionError("zero count accepted")
