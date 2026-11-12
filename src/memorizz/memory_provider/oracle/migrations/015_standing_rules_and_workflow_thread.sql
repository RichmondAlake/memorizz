-- Standing rules on a persona (personas), and the conversation a run came
-- from (workflow_memory).
--
--   personas.standing_rules           JSON array of the rules a named owner
--                                     approved: the tier between guardrails
--                                     and goals
--   workflow_memory.origin_thread_id  the conversation the run was made in.
--                                     Tool learning reads it to find a
--                                     request the user had to make twice
--
-- Additive and rerunnable: no stored data is rewritten. OracleProvider adds
-- the same columns at startup when they are missing; run this file for
-- schemas that are migrated out of band. Existing personas read as having no
-- standing rules, and existing runs as having no recorded conversation (so
-- they give tool learning calling errors and routing misses, but no re-asked
-- requests). Rule proposals waiting for an owner live in the agent's
-- persona_learning_state, and each approved change is an entry in the
-- existing evolution_history, so neither needs a column.

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
        'personas', 'standing_rules', 'CLOB CHECK (standing_rules IS JSON)'
    );
    add_column_if_missing('workflow_memory', 'origin_thread_id', 'VARCHAR2(255)');
END;
/

COMMIT;
