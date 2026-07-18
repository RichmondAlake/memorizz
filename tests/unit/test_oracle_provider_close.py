from memorizz.memory_provider.oracle import OracleProvider


class _Pool:
    def __init__(self):
        self.force = None

    def close(self, *, force=False):
        self.force = force


def test_close_force_closes_pool_and_is_idempotent():
    provider = OracleProvider.__new__(OracleProvider)
    pool = _Pool()
    provider.pool = pool

    provider.close()
    provider.close()

    assert pool.force is True
    assert provider.pool is None
