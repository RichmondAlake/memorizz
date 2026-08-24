Proposal argument hashes use canonical JSON with sorted keys and compact
separators, then SHA-256. approve requires a non-empty approver_id, accepts only
pending proposals, and expires when now >= expires_at. Expiry sets
status="expired" and raises ApprovalExpired. consume requires an approved,
unexpired proposal and an exact canonical argument hash; a mismatch raises
ArgumentMismatch. A successful consume sets status="consumed" before returning,
so every later consume raises ApprovalStateError. Keep the existing public
exception and Store APIs.
