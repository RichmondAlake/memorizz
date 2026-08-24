def ratio(total: float, count: int) -> float:
    """Return total/count; count must be positive."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    return total / count
