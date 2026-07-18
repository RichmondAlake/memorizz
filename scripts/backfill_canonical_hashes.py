#!/usr/bin/env python3
# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Backfill ``canonical_hash`` onto existing workflow_memory documents.

Idempotent (documents that already carry a hash are skipped), batched, and
dry-run-able. Run this before enabling continual learning on any existing
deployment so trajectory counts start from full history instead of zero.

Usage:
    # MongoDB (reads MONGODB_URI, or pass --mongodb-uri)
    python scripts/backfill_canonical_hashes.py --provider mongodb --dry-run
    python scripts/backfill_canonical_hashes.py --provider mongodb

    # Filesystem provider
    python scripts/backfill_canonical_hashes.py --provider filesystem \
        --root ~/.memorizz
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.long_term.procedural.workflow.canonicalization import (  # noqa: E402
    canonical_hash,
    canonical_signature,
)


def build_provider(args):
    if args.provider == "mongodb":
        from memorizz.memory_provider.mongodb.provider import (
            MongoDBConfig,
            MongoDBProvider,
        )

        uri = args.mongodb_uri or os.getenv("MONGODB_URI")
        if not uri:
            raise SystemExit("Set MONGODB_URI or pass --mongodb-uri.")
        return MongoDBProvider(MongoDBConfig(uri=uri))

    from memorizz.memory_provider.filesystem.provider import (
        FileSystemConfig,
        FileSystemProvider,
    )

    root = Path(args.root).expanduser()
    return FileSystemProvider(FileSystemConfig(root_path=root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("mongodb", "filesystem"), default="filesystem"
    )
    parser.add_argument("--mongodb-uri", default=None)
    parser.add_argument("--root", default="~/.memorizz")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args()

    provider = build_provider(args)
    documents = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY) or []

    scanned = updated = skipped = empty = 0
    pending = []
    for doc in documents:
        scanned += 1
        if not isinstance(doc, dict):
            continue
        if doc.get("canonical_hash"):
            skipped += 1
            continue
        signature = canonical_signature(doc.get("steps") or {})
        doc_hash = canonical_hash(signature)
        if not doc_hash:
            empty += 1
            continue
        # Provider-level _id first: MongoDB's update_by_id only accepts
        # ObjectIds, and every provider returns _id from list_all.
        doc_id = doc.get("_id") or doc.get("workflow_id")
        if not doc_id:
            continue
        pending.append(
            (
                str(doc_id),
                {
                    "canonical_hash": doc_hash,
                    "canonical_signature": signature,
                    "step_count": len(signature),
                },
            )
        )
        if len(pending) >= args.batch_size:
            updated += flush(provider, pending, args.dry_run)
            pending = []

    updated += flush(provider, pending, args.dry_run)

    action = "would update" if args.dry_run else "updated"
    print(
        f"scanned={scanned} {action}={updated} already_hashed={skipped} "
        f"no_steps={empty}"
    )
    return 0


def flush(provider, pending, dry_run: bool) -> int:
    if dry_run:
        return len(pending)
    written = 0
    for doc_id, patch in pending:
        try:
            if provider.update_by_id(
                doc_id, patch, memory_store_type=MemoryType.WORKFLOW_MEMORY
            ):
                written += 1
        except Exception as exc:  # keep going — idempotent re-runs pick it up
            print(f"  update failed for {doc_id}: {exc}", file=sys.stderr)
    return written


if __name__ == "__main__":
    raise SystemExit(main())
