"""Per-agent forgetting overrides ride along with the persisted retrieval policy."""

from __future__ import annotations

import pytest

from memorizz.retrieval import RetrievalPolicy


@pytest.mark.unit
def test_policy_carries_scoring_and_retention_overrides():
    policy = RetrievalPolicy.from_value(
        {
            "scoring": {"alpha_importance": 2, "recency_anchor": "created"},
            "retention": {"min_retention": 0.2},
        }
    )
    assert policy.scoring == {"alpha_importance": 2, "recency_anchor": "created"}
    assert policy.retention == {"min_retention": 0.2}
    assert RetrievalPolicy.from_value(policy.to_dict()) == policy
    assert "scoring" not in RetrievalPolicy().to_dict()


@pytest.mark.unit
def test_policy_rejects_non_mapping_overrides():
    with pytest.raises(TypeError):
        RetrievalPolicy(scoring="fast")
