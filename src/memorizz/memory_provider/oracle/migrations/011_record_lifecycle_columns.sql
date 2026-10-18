-- Columns that let records keep what used to be dropped on Oracle:
--   knowledge_base.status / superseded_by / superseded_at: the supersede
--     lifecycle memorizz_update_memory writes (it failed with
--     supersede_unsupported because update_by_id had nowhere to put it);
--   short_term_memory.metadata: JSON with the record's fields that have no
--     column of their own (knowledge_base.metadata already holds these);
--   semantic_cache.last_accessed: when a cached answer was last served.
--
-- Additive and rerunnable. OracleProvider applies the same columns at
-- startup when they are missing; run this file for schemas that are
-- migrated out of band. Existing rows keep NULLs.

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
    add_column_if_missing('knowledge_base', 'status', 'VARCHAR2(32)');
    add_column_if_missing('knowledge_base', 'superseded_by', 'VARCHAR2(255)');
    add_column_if_missing('knowledge_base', 'superseded_at', 'TIMESTAMP');
    add_column_if_missing('short_term_memory', 'metadata', 'CLOB');
    add_column_if_missing('semantic_cache', 'last_accessed', 'TIMESTAMP');
END;
/

COMMIT;
