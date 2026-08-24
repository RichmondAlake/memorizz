Keep ratio(total, count) as the public API. total must be a built-in int or
float, but never bool; wrong types raise TypeError and non-finite values raise
ValueError. count must be a built-in int, but never bool; wrong types raise
TypeError and values below one raise ValueError. Successful calls always return
float. Verification must still execute under python -O, so do not use bare
assert statements.
