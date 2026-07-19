-- ==============================================================================
-- Migration 003: Persist passive SHADOW-skill evaluations on workflows
-- ==============================================================================
-- The workflow row is the auditable source of truth. Skill-level shadow
-- counters are derived from this JSON evidence and remain advisory.
--
-- Execute as the Memorizz application schema owner:
--   sqlplus MEMORIZZ/<password>@<service> @003_add_shadow_evaluations.sql
-- ==============================================================================

ALTER TABLE workflow_memory ADD (
    shadow_evaluations CLOB
);

ALTER TABLE workflow_memory ADD CONSTRAINT chk_workflow_shadow_evaluations
    CHECK (shadow_evaluations IS JSON);

COMMIT;
