-- ==============================================================================
-- Migration 004: MemoRizz 0.5 production-governance and compaction parity
-- ==============================================================================
-- Run once against an existing pre-0.5 MemoRizz schema before deploying 0.5.
-- The provider also performs additive startup checks, but this migration gives
-- database operators a reviewable, change-controlled upgrade path.
--
-- Execute as the MemoRizz application schema owner:
--   sqlplus MEMORIZZ/<password>@<service> @004_production_governance_050.sql
-- ==============================================================================

-- Complete Toolbox JSON Schema and trusted-rebinding metadata.
ALTER TABLE toolbox ADD (input_schema CLOB);
ALTER TABLE toolbox ADD (tool_policy CLOB);
ALTER TABLE toolbox ADD (aliases CLOB);
ALTER TABLE toolbox ADD (deprecated_arguments CLOB);
ALTER TABLE toolbox ADD (queries CLOB);
ALTER TABLE toolbox ADD (import_reference VARCHAR2(1000));
ALTER TABLE toolbox ADD (user_id VARCHAR2(255));
ALTER TABLE toolbox ADD CONSTRAINT chk_toolbox_input_schema_json
    CHECK (input_schema IS JSON);
ALTER TABLE toolbox ADD CONSTRAINT chk_toolbox_policy_json
    CHECK (tool_policy IS JSON);
ALTER TABLE toolbox ADD CONSTRAINT chk_toolbox_aliases_json
    CHECK (aliases IS JSON);
ALTER TABLE toolbox ADD CONSTRAINT chk_toolbox_deprecated_args_json
    CHECK (deprecated_arguments IS JSON);
ALTER TABLE toolbox ADD CONSTRAINT chk_toolbox_queries_json
    CHECK (queries IS JSON);
CREATE INDEX idx_toolbox_user_id ON toolbox(user_id);

-- Canonical summary projection plus an explicit conversation compaction marker.
ALTER TABLE conversation_memory ADD (summary_id VARCHAR2(255));
CREATE INDEX idx_conv_summary_id ON conversation_memory(summary_id);

ALTER TABLE summaries ADD (source_message_ids CLOB);
ALTER TABLE summaries ADD (period_start NUMBER);
ALTER TABLE summaries ADD (period_end NUMBER);
ALTER TABLE summaries ADD (memory_units_count NUMBER(10) DEFAULT 0);
ALTER TABLE summaries ADD CONSTRAINT chk_summary_source_ids_json
    CHECK (source_message_ids IS JSON);

UPDATE summaries
SET source_message_ids = original_memory_ids
WHERE source_message_ids IS NULL
  AND original_memory_ids IS NOT NULL;

CREATE TABLE summary_message_links (
    summary_id VARCHAR2(255) NOT NULL,
    message_id RAW(16) NOT NULL,
    position NUMBER(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (summary_id, message_id),
    CONSTRAINT fk_summary_link_summary FOREIGN KEY (summary_id)
        REFERENCES summaries(summary_id) ON DELETE CASCADE,
    CONSTRAINT fk_summary_link_message FOREIGN KEY (message_id)
        REFERENCES conversation_memory(id) ON DELETE CASCADE
);
CREATE INDEX idx_summary_links_message ON summary_message_links(message_id);

-- Full semantic-cache scope/fingerprint/freshness metadata projection.
ALTER TABLE semantic_cache ADD (metadata CLOB);
ALTER TABLE semantic_cache ADD CONSTRAINT chk_cache_metadata_json
    CHECK (metadata IS JSON);

COMMIT;
