"""Native MongoDB observability spans, resource indexes and summary queries."""

from .index import SpanIndex, digest, resource_hash, validate_filters


class MongoSpanIndex(SpanIndex):
    provider_name = "mongodb"

    def __init__(self, db):
        self.db = db
        self.spans = db["observability_spans"]
        self.bundles = db["observability_bundles"]
        self.previews = db["observability_previews"]
        self.state = db["observability_state"]

    def ready(self):
        return bool(self.state.find_one({"_id": "schema_version", "value": 1}))

    def initialize(self):
        self.spans.create_index([("timestamp", -1), ("key", -1)], name="obs_time")
        self.spans.create_index("bundle_key", name="obs_bundle_key")
        for name, fields in (
            ("agent_thread", [("metadata.agent_id", 1), ("metadata.thread_id", 1)]),
            ("root", [("metadata.root_trace_id", 1)]),
            ("tenant", [("user_key", 1), ("metadata.application_id", 1)]),
            ("kind_status", [("metadata.kind", 1), ("metadata.status", 1)]),
            ("resources", [("resource_hashes", 1)]),
            ("memory", [("metadata.memory_id", 1)]),
            ("job", [("metadata.job_ref", 1)]),
            ("error", [("metadata.error_code", 1)]),
        ):
            self.spans.create_index(
                [*fields, ("timestamp", -1), ("key", -1)], name="obs_" + name
            )
        self.previews.create_index("timestamp", name="obs_preview_time")
        self.bundles.create_index("source_record_id", name="obs_source_record")
        self.state.update_one(
            {"_id": "schema_version"}, {"$setOnInsert": {"value": 1}}, upsert=True
        )
        return self.capabilities()

    def _require_ready(self):
        if not self.ready():
            raise RuntimeError(
                "Observability index is not initialized; run the explicit migration"
            )

    def put(self, summary, records):
        self._require_ready()
        existing = self.bundles.find_one({"_id": summary["bundle_key"]})
        if existing:
            if existing["fingerprint"] != summary["fingerprint"]:
                raise ValueError(
                    "Immutable bundle changed; record a new event or turn ID"
                )
            self.state.delete_one({"_id": "pending:" + summary["bundle_key"]})
            return
        self.state.update_one(
            {"_id": "pending:" + summary["bundle_key"]},
            {"$set": {"value": "pending"}},
            upsert=True,
        )
        # Per-event replacement is atomic and retry-safe on standalone MongoDB
        # too. The bundle checkpoint is written only after every child succeeds.
        for row in records:
            document = {key: value for key, value in row.items() if key != "preview"}
            self.spans.replace_one(
                {"_id": row["key"]}, {"_id": row["key"], **document}, upsert=True
            )
            if row["preview"]:
                self.previews.replace_one(
                    {"_id": row["key"]},
                    {
                        "_id": row["key"],
                        "timestamp": row["timestamp"],
                        "preview": row["preview"],
                    },
                    upsert=True,
                )
            else:
                self.previews.delete_one({"_id": row["key"]})
        self.bundles.replace_one(
            {"_id": summary["bundle_key"]},
            {"_id": summary["bundle_key"], **summary},
            upsert=True,
        )
        self.state.delete_one({"_id": "pending:" + summary["bundle_key"]})

    def pending_count(self):
        return self.state.count_documents({"value": "pending"})

    def _predicate(self, filters, boundary=None):
        clauses = []
        identities = []
        for key, field in (("agent_ids", "agent_id"), ("memory_ids", "memory_id")):
            if filters.get(key):
                identities.append({"metadata." + field: {"$in": filters[key]}})
        if identities:
            clauses.append({"$or": identities})
        for key in (
            "application_id",
            "thread_id",
            "root_trace_id",
            "run_id",
            "turn_id",
        ):
            if filters.get(key) is not None:
                clauses.append({"metadata." + key: filters[key]})
        if "user_id" in filters:
            clauses.append({"user_key": digest(filters["user_id"])})
        for key, column in (("event_kinds", "kind"), ("statuses", "status")):
            if filters.get(key):
                clauses.append({"metadata." + column: {"$in": filters[key]}})
        if filters.get("tool_name"):
            clauses.append(
                {
                    "$or": [
                        {"metadata.logical_tool_name": filters["tool_name"]},
                        {"metadata.tool_name": filters["tool_name"]},
                    ]
                }
            )
        if filters.get("success") is not None:
            clauses.append(
                {
                    "metadata.status": {
                        "$in": ["success"]
                        if filters["success"]
                        else ["error", "failed"]
                    }
                }
            )
        for key, operator in (("start_time", "$gte"), ("end_time", "$lte")):
            if filters.get(key):
                clauses.append({"timestamp": {operator: filters[key]}})
        if filters.get("resource_refs"):
            clauses.append(
                {
                    "resource_hashes": {
                        "$in": [resource_hash(v) for v in filters["resource_refs"]]
                    }
                }
            )
        if filters.get("query"):
            query = filters["query"]
            clauses.append(
                {
                    "$or": [
                        *(
                            {"metadata." + field: query}
                            for field in (
                                "agent_id",
                                "thread_id",
                                "root_trace_id",
                                "run_id",
                                "turn_id",
                                "job_ref",
                                "error_code",
                            )
                        ),
                        {"resource_hashes": resource_hash(query)},
                        {"user_key": digest(query)},
                    ]
                }
            )
        if boundary:
            clauses.append(
                {
                    "$or": [
                        {"timestamp": {"$lt": boundary[0]}},
                        {"timestamp": boundary[0], "key": {"$lt": boundary[1]}},
                    ]
                }
            )
        return {"$and": clauses} if clauses else {}

    def select(self, filters, *, boundary=None, limit=251):
        self._require_ready()
        return list(
            self.spans.find(self._predicate(filters, boundary), {"_id": 0})
            .sort([("timestamp", -1), ("key", -1)])
            .limit(limit)
        )

    def summaries(self, *, limit=250, **filters):
        self._require_ready()
        rows = self.spans.aggregate(
            [
                {"$match": self._predicate(validate_filters(filters))},
                {
                    "$group": {
                        "_id": {
                            "agent_id": "$metadata.agent_id",
                            "thread_id": "$metadata.thread_id",
                        },
                        "event_count": {"$sum": 1},
                        "bundles": {"$addToSet": "$bundle_key"},
                        "latest_timestamp": {"$max": "$timestamp"},
                    }
                },
                {"$sort": {"latest_timestamp": -1}},
                {"$limit": max(1, min(int(limit), 1000))},
            ]
        )
        return [
            {
                **row["_id"],
                "event_count": row["event_count"],
                "bundle_count": len(row["bundles"]),
                "latest_timestamp": row["latest_timestamp"],
            }
            for row in rows
        ]

    def preview(self, event_id, **filters):
        self._require_ready()
        query = {
            "$and": [
                self._predicate(validate_filters(filters)),
                {"metadata.event_id": event_id},
            ]
        }
        span = self.spans.find_one(query, {"key": 1})
        row = self.previews.find_one({"_id": span["key"]}) if span else None
        return row["preview"] if row else None

    def retention(self, *, dry_run=True, **policy):
        self._require_ready()
        cutoffs = self.retention_cutoffs(**policy)
        verified = {"metadata.verified": True, "metadata.kind": "verified_outcome"}
        query = {
            "$or": [
                {**verified, "timestamp": {"$lt": cutoffs["outcomes"]}},
                {"$nor": [verified], "timestamp": {"$lt": cutoffs["metadata"]}},
            ]
        }
        count = self.spans.count_documents(query)
        previews = self.previews.count_documents(
            {"timestamp": {"$lt": cutoffs["previews"]}}
        )
        if not dry_run:
            # Bounded batches; never materialize every historical event ID.
            while True:
                keys = [
                    row["_id"] for row in self.spans.find(query, {"_id": 1}).limit(500)
                ]
                if not keys:
                    break
                self.previews.delete_many({"_id": {"$in": keys}})
                self.spans.delete_many({"_id": {"$in": keys}})
            self.previews.delete_many({"timestamp": {"$lt": cutoffs["previews"]}})
            # Bound memory use while removing expired orphan checkpoints.
            for row in self.bundles.find(
                {"timestamp": {"$lt": cutoffs["metadata"]}}, {"_id": 1}
            ):
                if not self.spans.find_one({"bundle_key": row["_id"]}, {"_id": 1}):
                    self.bundles.delete_one({"_id": row["_id"]})
        return {
            "dry_run": dry_run,
            "metadata_events": count,
            "previews": previews,
            "cutoffs": cutoffs,
            "source_bundles_unchanged": True,
        }
