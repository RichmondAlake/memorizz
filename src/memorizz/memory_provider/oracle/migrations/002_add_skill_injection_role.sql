-- ==============================================================================
-- Migration 002: Persist learned-skill instruction authority
-- ==============================================================================
-- Existing skills retain the pre-upgrade behavior: user-context injection.
-- New reviewed skills may opt into developer/application authority.
--
-- Execute as the Memorizz application schema owner:
--   sqlplus MEMORIZZ/<password>@<service> @002_add_skill_injection_role.sql
-- ==============================================================================

ALTER TABLE skillbox ADD (
    injection_role VARCHAR2(20) DEFAULT 'user' NOT NULL
);

ALTER TABLE skillbox ADD CONSTRAINT chk_skillbox_injection_role
    CHECK (injection_role IN ('user', 'developer'));

COMMIT;
