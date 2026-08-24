import uuid

import pytest


class FakeStore:
    """In-memory automation store for testing."""

    def __init__(self):
        self.created_jobs = {}

    def create_job(self, job):
        self.created_jobs[job.job_id] = job
        return job

    def get_job(self, job_id):
        return self.created_jobs.get(job_id)

    def list_jobs(self, agent_id=None, enabled=None):
        jobs = list(self.created_jobs.values())
        if agent_id is not None:
            jobs = [j for j in jobs if j.agent_id == agent_id]
        if enabled is not None:
            jobs = [j for j in jobs if j.enabled == enabled]
        return jobs

    def pause_job(self, job_id):
        job = self.created_jobs[job_id]
        job.enabled = False
        return job

    def resume_job(self, job_id):
        job = self.created_jobs[job_id]
        job.enabled = True
        return job

    def delete_job(self, job_id):
        return bool(self.created_jobs.pop(job_id, None))

    def claim_due_jobs(self, worker_id, now_utc, limit, lease_seconds):
        return []

    def start_run(self, job, scheduled_for, worker_id):
        raise NotImplementedError

    def finish_run(
        self, run_id, status, error, result_summary, result_payload, attempt=1
    ):
        raise NotImplementedError

    def record_delivery(self, run_id, delivery):
        raise NotImplementedError

    def list_runs(self, job_id, limit=50):
        return []

    def update_job(self, job_id, patch):
        job = self.created_jobs[job_id]
        for k, v in patch.items():
            setattr(job, k, v)
        return job


def _patch_store(monkeypatch, fake_store=None):
    """Patch the automation store factory to return a FakeStore."""
    store = fake_store or FakeStore()
    monkeypatch.setattr(
        "memorizz.memagent.managers.automation_manager.get_automation_store",
        lambda _provider: store,
    )
    return store


def test_memagent_registers_automation_tools_when_store_available(monkeypatch):
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object())
    tool_names = set(agent.tool_manager.list_tools())
    assert "automation_create_job" in tool_names
    assert "automation_list_jobs" in tool_names
    assert "automation_pause_job" in tool_names
    assert "automation_resume_job" in tool_names
    assert "automation_delete_job" in tool_names
    assert "automation_run_now" in tool_names

    tool_fn = agent.tool_manager.tools["automation_create_job"]["function"]
    resp = tool_fn(
        name="Daily brief",
        schedule_type="interval",
        interval_seconds=60,
        timezone="UTC",
        query_template="Hello {today_iso}",
        memory_id=str(uuid.uuid4()),
        whatsapp_to=[],
    )
    assert resp["ok"] is True
    assert resp["created"] is True
    schema = agent.tool_manager.tools["automation_create_job"]["metadata"][
        "input_schema"
    ]
    assert "confirm" not in schema["properties"]
    assert (
        agent.tool_manager.get_tool_policy("automation_create_job")["requires_approval"]
        is True
    )


# ---------------------------------------------------------------------------
# SDK convenience method tests
# ---------------------------------------------------------------------------


def test_create_automation_sdk(monkeypatch):
    """SDK create_automation returns an AutomationJob with correct fields."""
    from memorizz.automation import AutomationJob
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    assert agent.has_automations()

    job = agent.create_automation(
        name="Test Job",
        schedule_type="interval",
        query_template="Hello {today_iso}",
        interval_seconds=300,
    )
    assert isinstance(job, AutomationJob)
    assert job.name == "Test Job"
    assert job.schedule_type == "interval"
    assert job.interval_seconds == 300
    assert job.enabled is True
    assert job.agent_id == agent.agent_id
    assert job.action_config["query_template"] == "Hello {today_iso}"
    assert job.action_config.get("memory_id")  # auto-generated


def test_create_automation_cron_sdk(monkeypatch):
    """SDK create_automation with cron schedule."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Daily Brief",
        schedule_type="cron",
        cron_expr="0 9 * * *",
        query_template="Good morning",
    )
    assert job.schedule_type == "cron"
    assert job.cron_expr == "0 9 * * *"


def test_create_automation_with_whatsapp(monkeypatch):
    """SDK create_automation with WhatsApp delivery."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Alert",
        schedule_type="interval",
        interval_seconds=60,
        query_template="Check status",
        whatsapp_to=["+15551234567"],
    )
    assert job.delivery_type == "whatsapp_twilio"
    assert "whatsapp:+15551234567" in job.delivery_config["whatsapp_to"]


def test_create_automation_raises_when_unavailable():
    """SDK methods raise ValueError when automations are not configured."""
    from memorizz.memagent import MemAgent

    agent = MemAgent(memory_provider=False)
    assert not agent.has_automations()

    with pytest.raises(ValueError, match="not available"):
        agent.create_automation(
            name="Test",
            schedule_type="interval",
            query_template="Hello",
            interval_seconds=60,
            timezone="UTC",
        )


def test_create_automation_raises_on_missing_timezone(monkeypatch):
    """create_automation raises ValueError when no timezone can be resolved."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)
    monkeypatch.delenv("MEMORIZZ_DEFAULT_TIMEZONE", raising=False)

    agent = MemAgent(memory_provider=object())
    with pytest.raises(ValueError, match="timezone is required"):
        agent.create_automation(
            name="Test",
            schedule_type="interval",
            query_template="Hello",
            interval_seconds=60,
        )


def test_create_automation_timezone_cascade(monkeypatch):
    """Timezone resolution: explicit > agent default > env var."""
    from memorizz.memagent import MemAgent

    # 1. Explicit timezone wins
    _patch_store(monkeypatch)
    agent = MemAgent(memory_provider=object(), default_timezone="US/Eastern")
    job = agent.create_automation(
        name="T1",
        schedule_type="one_shot",
        query_template="Hi",
        timezone="Europe/London",
    )
    assert job.timezone == "Europe/London"

    # 2. Falls back to agent default
    _patch_store(monkeypatch)
    agent2 = MemAgent(memory_provider=object(), default_timezone="US/Pacific")
    job2 = agent2.create_automation(
        name="T2",
        schedule_type="one_shot",
        query_template="Hi",
    )
    assert job2.timezone == "US/Pacific"

    # 3. Falls back to env var
    _patch_store(monkeypatch)
    monkeypatch.setenv("MEMORIZZ_DEFAULT_TIMEZONE", "Asia/Tokyo")
    agent3 = MemAgent(memory_provider=object())
    job3 = agent3.create_automation(
        name="T3",
        schedule_type="one_shot",
        query_template="Hi",
    )
    assert job3.timezone == "Asia/Tokyo"


def test_list_automations_sdk(monkeypatch):
    """SDK list_automations returns jobs for this agent."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    agent.create_automation(
        name="Job 1",
        schedule_type="interval",
        query_template="Hello",
        interval_seconds=60,
    )
    agent.create_automation(
        name="Job 2",
        schedule_type="interval",
        query_template="World",
        interval_seconds=120,
    )
    jobs = agent.list_automations()
    assert len(jobs) == 2
    names = {j.name for j in jobs}
    assert names == {"Job 1", "Job 2"}


def test_get_automation_sdk(monkeypatch):
    """SDK get_automation returns a job or None."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Lookup Test",
        schedule_type="one_shot",
        query_template="Hi",
    )

    found = agent.get_automation(job.job_id)
    assert found is not None
    assert found.job_id == job.job_id

    missing = agent.get_automation("nonexistent-id")
    assert missing is None


def test_pause_resume_automation_sdk(monkeypatch):
    """SDK pause/resume toggle the enabled state."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Toggle Test",
        schedule_type="interval",
        query_template="Hi",
        interval_seconds=60,
    )
    assert job.enabled is True

    paused = agent.pause_automation(job.job_id)
    assert paused.enabled is False

    resumed = agent.resume_automation(job.job_id)
    assert resumed.enabled is True


def test_delete_automation_sdk(monkeypatch):
    """SDK delete_automation returns True for existing, False for missing."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Delete Test",
        schedule_type="one_shot",
        query_template="Hi",
    )

    assert agent.delete_automation(job.job_id) is True
    assert agent.delete_automation(job.job_id) is False


def test_trigger_automation_sdk(monkeypatch):
    """SDK trigger_automation sets next_run_at to approximately now."""
    from datetime import datetime, timezone

    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Trigger Test",
        schedule_type="interval",
        query_template="Hi",
        interval_seconds=86400,
    )

    before = datetime.now(timezone.utc)
    triggered = agent.trigger_automation(job.job_id)
    after = datetime.now(timezone.utc)

    assert triggered.enabled is True
    assert before <= triggered.next_run_at <= after


def test_list_automation_runs_sdk(monkeypatch):
    """SDK list_automation_runs delegates to store."""
    from memorizz.memagent import MemAgent

    _patch_store(monkeypatch)

    agent = MemAgent(memory_provider=object(), default_timezone="UTC")
    job = agent.create_automation(
        name="Runs Test",
        schedule_type="one_shot",
        query_template="Hi",
    )

    runs = agent.list_automation_runs(job.job_id)
    assert runs == []  # FakeStore returns empty list
