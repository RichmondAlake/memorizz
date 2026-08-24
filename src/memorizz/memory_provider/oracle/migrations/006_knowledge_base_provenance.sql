-- ==============================================================================
-- Migration 006: provider-neutral knowledge-base provenance
-- ==============================================================================
-- Adds the fields used by source-grounded retrieval and multi-source semantic
-- memories. Existing rows remain valid and return empty provenance values.

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
    add_column_if_missing('knowledge_base', 'source_id', 'VARCHAR2(512)');
    add_column_if_missing('knowledge_base', 'parent_source_id', 'VARCHAR2(512)');
    add_column_if_missing('knowledge_base', 'linked_source_ids', 'CLOB');
    add_column_if_missing('knowledge_base', 'metadata', 'CLOB');
END;
/

COMMIT;
