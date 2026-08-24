class Cache:
    def __init__(self):
        self._data = {}

    @staticmethod
    def _key(query, user_id, domain, data_version):
        return query.strip().casefold()

    def set(self, query, value, *, user_id, domain, data_version, ttl, now):
        key = self._key(query, user_id, domain, data_version)
        self._data[key] = (value, now + ttl)

    def get(self, query, *, user_id, domain, data_version, now):
        key = self._key(query, user_id, domain, data_version)
        item = self._data.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at < now:
            return None
        return value

    def invalidate(self, *, domain=None, data_version=None):
        self._data.clear()

    def __len__(self):
        return len(self._data)
