-- Verified outcomes and the skill holdout (workflow_memory), and persona
-- guardrails (personas).
--
--   workflow_memory.outcome_verified  1 when a host or the outcome evaluator
--                                     confirmed the run's outcome
--   workflow_memory.skills_held_out   JSON array of the active skills that
--                                     matched the run and were withheld
--   personas.guardrails               JSON array of the rules learning
--                                     cannot write
--
-- Additive and rerunnable: no stored data is rewritten. OracleProvider adds
-- the same columns at startup when they are missing; run this file for
-- schemas that are migrated out of band. Existing rows read as not
-- verified, with nothing held out and no guardrails. Entity history dates,
-- source levels and aliases live inside the existing attributes and
-- metadata JSON, so entity_memory needs no change.

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
    add_column_if_missing('workflow_memory', 'outcome_verified', 'NUMBER(1) DEFAULT 0');
    add_column_if_missing(
        'workflow_memory', 'skills_held_out', 'CLOB CHECK (skills_held_out IS JSON)'
    );
    add_column_if_missing('personas', 'guardrails', 'CLOB CHECK (guardrails IS JSON)');
END;
/

COMMIT;
