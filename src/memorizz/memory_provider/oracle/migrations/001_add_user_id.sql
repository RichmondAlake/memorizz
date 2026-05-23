-- ==============================================================================
-- Migration 001: Add user_id columns for multi-tenant scoping
-- ==============================================================================
-- Run this ONCE against an existing Memorizz Oracle schema to add multi-tenant
-- support. Safe to run before upgrading application code because:
--   1. All new columns are NULLable.
--   2. Existing rows get user_id = NULL and remain visible to "anonymous" scope.
--
-- For single-tenant installations that do not pass user_id, behavior is
-- unchanged.
--
-- Execute as the memorizz application schema owner:
--   sqlplus MEMORIZZ/<password>@<service> @001_add_user_id.sql
-- ==============================================================================

ALTER TABLE conversation_memory ADD (user_id VARCHAR2(255));
CREATE INDEX idx_conv_user_id ON conversation_memory(user_id);
CREATE INDEX idx_conv_memory_user ON conversation_memory(memory_id, user_id);

ALTER TABLE knowledge_base ADD (user_id VARCHAR2(255));
CREATE INDEX idx_ltm_user_id ON knowledge_base(user_id);

ALTER TABLE short_term_memory ADD (user_id VARCHAR2(255));
CREATE INDEX idx_stm_user_id ON short_term_memory(user_id);

ALTER TABLE workflow_memory ADD (user_id VARCHAR2(255));
CREATE INDEX idx_workflow_user_id ON workflow_memory(user_id);

ALTER TABLE summaries ADD (user_id VARCHAR2(255));
CREATE INDEX idx_summaries_user_id ON summaries(user_id);

-- semantic_cache historically lacked memory_id and session_id; add them here so
-- the cache's tenant and session scopes can be enforced server-side.
ALTER TABLE semantic_cache ADD (memory_id VARCHAR2(255));
ALTER TABLE semantic_cache ADD (session_id VARCHAR2(255));
ALTER TABLE semantic_cache ADD (user_id VARCHAR2(255));
CREATE INDEX idx_cache_user_id ON semantic_cache(user_id);

ALTER TABLE entity_memory ADD (user_id VARCHAR2(255));
CREATE INDEX idx_entity_memory_user_id ON entity_memory(user_id);

ALTER TABLE tool_log ADD (user_id VARCHAR2(255));
CREATE INDEX idx_tool_log_user_id ON tool_log(user_id);

COMMIT;
