import hashlib
import json
from dataclasses import dataclass


class ApprovalError(Exception):
    pass


class ApprovalExpired(ApprovalError):
    pass


class ApprovalStateError(ApprovalError):
    pass


class ArgumentMismatch(ApprovalError):
    pass


@dataclass
class Proposal:
    proposal_id: str
    argument_hash: str
    expires_at: float
    status: str = "pending"
    approver_id: str | None = None


class Store:
    def __init__(self):
        self._items = {}

    @staticmethod
    def hash_arguments(arguments):
        raw = json.dumps(arguments).encode()
        return hashlib.sha256(raw).hexdigest()

    def propose(self, proposal_id, arguments, *, now, ttl):
        item = Proposal(
            proposal_id,
            self.hash_arguments(arguments),
            now + ttl,
        )
        self._items[proposal_id] = item
        return item

    def approve(self, proposal_id, *, approver_id, now):
        item = self._items[proposal_id]
        item.status = "approved"
        item.approver_id = approver_id
        return item

    def consume(self, proposal_id, arguments, *, now):
        item = self._items[proposal_id]
        if item.status != "approved":
            raise ApprovalStateError("not approved")
        if item.argument_hash != self.hash_arguments(arguments):
            raise ArgumentMismatch("arguments changed")
        return item
