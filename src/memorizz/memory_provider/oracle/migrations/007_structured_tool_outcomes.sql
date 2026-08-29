-- Structured tool execution outcomes for Memorizz observability.
-- Existing rows remain successful/failed according to their legacy boolean.

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
    add_column_if_missing(
        'tool_log', 'outcome', 'VARCHAR2(32) DEFAULT ''success'''
    );
    add_column_if_missing('tool_log', 'outcome_details', 'CLOB');
END;
/

UPDATE tool_log
SET outcome = CASE WHEN success = 1 THEN 'success' ELSE 'error' END
WHERE outcome IS NULL OR outcome = 'success';

COMMIT;
