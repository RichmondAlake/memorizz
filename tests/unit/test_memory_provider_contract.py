"""Portable capability-contract checks for first-party memory providers."""

from __future__ import annotations

import pytest

from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


@pytest.mark.unit
def test_first_party_provider_capabilities_have_one_stable_shape(tmp_path):
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider
    from memorizz.memory_provider.oracle.provider import OracleProvider

    filesystem = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    try:
        reports = [
            filesystem.memory_capabilities().to_dict(),
            MongoDBProvider.__new__(MongoDBProvider).memory_capabilities().to_dict(),
            OracleProvider.__new__(OracleProvider).memory_capabilities().to_dict(),
        ]
    finally:
        filesystem.close()

    assert len({tuple(sorted(report)) for report in reports}) == 1
    assert reports[0]["batch_store"] is True
    assert reports[1]["batch_store"] is True
    assert reports[2]["batch_store"] is False
    assert all(report["scoped_search"] and report["provenance"] for report in reports)
