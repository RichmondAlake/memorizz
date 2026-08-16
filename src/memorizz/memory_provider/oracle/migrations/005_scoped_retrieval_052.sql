-- ==============================================================================
-- Migration 005: MemoRizz 0.5.2 exact thread/namespace retrieval scopes
-- ==============================================================================
-- Safe to run against both upgraded and freshly-created 0.5.x schemas. The
-- provider performs the same additive checks at startup; this file exists for
-- operators that require reviewable, change-controlled database migrations.

DECLARE
    PROCEDURE add_column_if_missing(
        table_name_in VARCHAR2,
        column_name_in VARCHAR2,
        definition_in VARCHAR2
    ) IS
        column_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO column_count
        FROM user_tab_columns
        WHERE table_name = UPPER(table_name_in)
          AND column_name = UPPER(column_name_in);

        IF column_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE ' || table_name_in || ' ADD (' ||
                column_name_in || ' ' || definition_in || ')';
        END IF;
    END;
BEGIN
    add_column_if_missing('summaries', 'thread_id', 'VARCHAR2(255)');

    add_column_if_missing('knowledge_base', 'knowledge_base_id', 'VARCHAR2(64)');
    add_column_if_missing('knowledge_base', 'namespace', 'VARCHAR2(255)');
    add_column_if_missing('knowledge_base', 'chunk_index', 'NUMBER(10) DEFAULT 0');
    add_column_if_missing('knowledge_base', 'chunk_count', 'NUMBER(10) DEFAULT 1');
    add_column_if_missing('knowledge_base', 'chunking_strategy', 'VARCHAR2(32)');
END;
/

-- Backfill a summary only when every linked source message belongs to one
-- non-null thread. Ambiguous legacy summaries deliberately remain unscoped.
DECLARE
    link_table_count NUMBER;
BEGIN
    SELECT COUNT(*) INTO link_table_count
    FROM user_tables
    WHERE table_name = 'SUMMARY_MESSAGE_LINKS';

    IF link_table_count > 0 THEN
        EXECUTE IMMEDIATE q'[
            MERGE INTO summaries summary
            USING (
                SELECT link.summary_id, MIN(message.thread_id) AS thread_id
                FROM summary_message_links link
                JOIN conversation_memory message ON message.id = link.message_id
                WHERE message.thread_id IS NOT NULL
                GROUP BY link.summary_id
                HAVING COUNT(DISTINCT message.thread_id) = 1
            ) source
            ON (summary.summary_id = source.summary_id)
            WHEN MATCHED THEN UPDATE
                SET summary.thread_id = source.thread_id
                WHERE summary.thread_id IS NULL
        ]';
    END IF;
END;
/

DECLARE
    PROCEDURE add_index_if_missing(
        index_name_in VARCHAR2,
        statement_in VARCHAR2
    ) IS
        index_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO index_count
        FROM user_indexes
        WHERE index_name = UPPER(index_name_in);

        IF index_count = 0 THEN
            BEGIN
                EXECUTE IMMEDIATE statement_in;
            EXCEPTION
                WHEN OTHERS THEN
                    -- ORA-01408 means an equivalent index already exists.
                    IF SQLCODE != -1408 THEN
                        RAISE;
                    END IF;
            END;
        END IF;
    END;
BEGIN
    add_index_if_missing(
        'idx_summaries_memory_thread',
        'CREATE INDEX idx_summaries_memory_thread ON summaries(memory_id, thread_id)'
    );
    add_index_if_missing(
        'idx_kb_namespace',
        'CREATE INDEX idx_kb_namespace ON knowledge_base(namespace)'
    );
END;
/

COMMIT;
