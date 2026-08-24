delays(attempts, base=0.1, cap=1.0) treats attempts as the total number of
operation attempts, including the initial attempt, so it returns exactly
attempts-1 retry delays. attempts must be a built-in int but not bool; wrong
types raise TypeError and values below one raise ValueError. base and cap must
be built-in int or float but not bool, finite, and strictly positive; wrong
types raise TypeError and invalid values raise ValueError. Every returned delay
is float and equals min(cap, base * 2**retry_index).
