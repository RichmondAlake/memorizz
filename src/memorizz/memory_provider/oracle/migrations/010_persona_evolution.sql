-- Persona evolution state for the personas table: goals (previously not
-- stored, so reloads returned the background as goals), version (bumped by
-- each Persona.update) and evolution_history, the JSON array describing
-- what changed and why.
--
-- Additive and rerunnable. OracleProvider applies the same columns at
-- startup when they are missing; run this file for schemas that are
-- migrated out of band. Existing rows keep NULL goals/history and take
-- version 1 from the column default.

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
    add_column_if_missing('personas', 'goals', 'CLOB');
    add_column_if_missing('personas', 'version', 'NUMBER(10) DEFAULT 1');
    add_column_if_missing(
        'personas', 'evolution_history', 'CLOB CHECK (evolution_history IS JSON)'
    );
END;
/

COMMIT;
