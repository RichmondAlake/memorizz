"""Synthetic host for visually exercising the real Persona Evolution UI."""
import os
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch

import uvicorn

from memorizz.ui.app import create_app


class Fixture:
    def __init__(self):
        self.profile = {
            "revision": 1,
            "paused": False,
            "pending": None,
            "reviews": [],
            "review_state": "not_reviewed",
            "history_limit": 50,
            "persona": {
                "name": "OpenSpeech",
                "role": "Learning partner",
                "version": 1,
                "goals": "Help this user understand their sources, learn effectively, and create grounded work.",
                "evolution_history": [],
            },
        }

    def view(self, user_id):
        assert user_id == "synthetic-learner"
        return deepcopy(self.profile)

    def reflect(self, user_id, *, actor_id):
        self.profile["revision"] += 1
        self.profile["last_review_at"] = "2026-09-07T10:30:00Z"
        self.profile["review_state"] = "proposed"
        self.profile["pending"] = {
            "id": "review-example-first",
            "reason": "You asked to start with a concrete example before introducing formal definitions.",
            "changes": {
                "goals": {
                    "old": self.profile["persona"]["goals"],
                    "new": "Explain new concepts through a practical example first, then connect it to the formal definition. Keep every source-based claim grounded in the user's material.",
                }
            },
            "evidence": [
                {
                    "id": "synthetic-memory-1",
                    "thread_id": "synthetic-thread",
                    "timestamp": "2026-09-07T09:00:00Z",
                    "text": "I learn much better when you start with a real example. Please do that before the formal definition.",
                }
            ],
        }
        return self.view(user_id)

    def decide(self, user_id, *, proposal_id, revision, accept, actor_id):
        assert (
            revision == self.profile["revision"]
            and proposal_id == self.profile["pending"]["id"]
        )
        pending = self.profile["pending"]
        if accept:
            self.profile["persona"]["goals"] = pending["changes"]["goals"]["new"]
            self.profile["persona"]["version"] += 1
            self.profile["persona"]["evolution_history"].append(
                {
                    "version": self.profile["persona"]["version"],
                    "timestamp": "2026-09-07T10:31:00Z",
                    "action": "applied",
                    "actor_id": actor_id,
                    "review_id": pending["id"],
                    "change_trigger": {"reason": pending["reason"]},
                    "changes": pending["changes"],
                    "evidence": pending["evidence"],
                }
            )
        self.profile["pending"] = None
        self.profile["revision"] += 1
        return self.view(user_id)

    def pause(self, user_id, *, revision, paused):
        assert revision == self.profile["revision"]
        self.profile["paused"] = paused
        self.profile["revision"] += 1
        return self.view(user_id)

    def undo(self, user_id, *, revision, actor_id):
        assert revision == self.profile["revision"]
        previous = self.profile["persona"]["evolution_history"][-1]
        self.profile["persona"]["goals"] = previous["changes"]["goals"]["old"]
        self.profile["persona"]["version"] += 1
        self.profile["persona"]["evolution_history"].append(
            {
                **previous,
                "version": self.profile["persona"]["version"],
                "action": "undo",
                "actor_id": actor_id,
                "changes": {
                    "goals": {
                        "old": previous["changes"]["goals"]["new"],
                        "new": previous["changes"]["goals"]["old"],
                    }
                },
                "change_trigger": {"reason": "Previous change undone."},
                "evidence": [],
            }
        )
        self.profile["revision"] += 1
        return self.view(user_id)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="memorizz-persona-browser-") as root:
        os.environ.update(
            MEMORIZZ_UI_AUTH_TOKEN="persona-browser-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="true",
            MEMORIZZ_UI_AUDIT_LOG=root + "/audit.jsonl",
        )
        with patch("memorizz.ui.app._load_layered_env", lambda: None):
            app = create_app(
                persona_evolution=Fixture(), persona_evolution_actions=True
            )
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=int(os.getenv("MEMORIZZ_PERSONA_TEST_PORT", "8783")),
            lifespan="off",
        )
