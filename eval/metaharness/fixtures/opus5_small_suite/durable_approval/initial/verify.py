from approval import Store

store = Store()
store.propose("p1", {"b": 2, "a": 1}, now=0, ttl=10)
store.approve("p1", approver_id="host", now=1)
store.consume("p1", {"a": 1, "b": 2}, now=2)
