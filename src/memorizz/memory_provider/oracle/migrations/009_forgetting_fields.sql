-- Generative-Agents forgetting fields for Memorizz primary memory records:
-- importance (0..1), last_accessed_at, access_count, a reversible
-- retention_state ('active' | 'suppressed') and retention_meta, the JSON
-- audit trail of who suppressed/unsuppressed a record and why.
--
-- Additive and rerunnable. OracleProvider applies the same columns at
-- startup when they are missing; run this file for schemas that are
-- migrated out of band. knowledge_base already carries importance and
-- access_count.

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
    add_column_if_missing('conversation_memory', 'importance', 'NUMBER(5,4)');
    add_column_if_missing('conversation_memory', 'last_accessed_at', 'TIMESTAMP');
    add_column_if_missing('conversation_memory', 'access_count', 'NUMBER(10) DEFAULT 0');
    add_column_if_missing(
        'conversation_memory', 'retention_state', 'VARCHAR2(32) DEFAULT ''active'''
    );
    add_column_if_missing(
        'conversation_memory', 'retention_meta', 'CLOB CHECK (retention_meta IS JSON)'
    );

    add_column_if_missing('knowledge_base', 'last_accessed_at', 'TIMESTAMP');
    add_column_if_missing(
        'knowledge_base', 'retention_state', 'VARCHAR2(32) DEFAULT ''active'''
    );
    add_column_if_missing(
        'knowledge_base', 'retention_meta', 'CLOB CHECK (retention_meta IS JSON)'
    );

    add_column_if_missing('entity_memory', 'importance', 'NUMBER(5,4)');
    add_column_if_missing('entity_memory', 'last_accessed_at', 'TIMESTAMP');
    add_column_if_missing('entity_memory', 'access_count', 'NUMBER(10) DEFAULT 0');
    add_column_if_missing(
        'entity_memory', 'retention_state', 'VARCHAR2(32) DEFAULT ''active'''
    );
    add_column_if_missing(
        'entity_memory', 'retention_meta', 'CLOB CHECK (retention_meta IS JSON)'
    );

    add_column_if_missing('summaries', 'importance', 'NUMBER(5,4)');
    add_column_if_missing('summaries', 'last_accessed_at', 'TIMESTAMP');
    add_column_if_missing('summaries', 'access_count', 'NUMBER(10) DEFAULT 0');
    add_column_if_missing(
        'summaries', 'retention_state', 'VARCHAR2(32) DEFAULT ''active'''
    );
    add_column_if_missing(
        'summaries', 'retention_meta', 'CLOB CHECK (retention_meta IS JSON)'
    );

    add_column_if_missing('workflow_memory', 'importance', 'NUMBER(5,4)');
    add_column_if_missing('workflow_memory', 'last_accessed_at', 'TIMESTAMP');
    add_column_if_missing('workflow_memory', 'access_count', 'NUMBER(10) DEFAULT 0');
    add_column_if_missing(
        'workflow_memory', 'retention_state', 'VARCHAR2(32) DEFAULT ''active'''
    );
    add_column_if_missing(
        'workflow_memory', 'retention_meta', 'CLOB CHECK (retention_meta IS JSON)'
    );
END;
/

COMMIT;
