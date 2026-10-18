-- Default every plain TIMESTAMP column of the MemoRizz tables to the UTC
-- clock, SYS_EXTRACT_UTC(SYSTIMESTAMP).
--
-- Schemas created before MemoRizz stored UTC default created_at, updated_at
-- and timestamp to CURRENT_TIMESTAMP, the session's local time
-- (python-oracledb thin mode sets the session time zone to the Python
-- process's UTC offset), while updates wrote the database server's clock. Now
-- every TIMESTAMP column holds UTC wall-clock time.
--
-- Rerunnable and safe: a new default only applies to rows inserted later, no
-- data changes. OracleProvider applies the same change at startup for the
-- columns that still need it; run this file for schemas that are migrated
-- out of band (or when the application user may not ALTER the tables).
-- Existing values are converted by 013_utc_timestamps.sql, which is run by
-- hand. TIMESTAMP WITH TIME ZONE columns (automation schedules) and tables
-- other than MemoRizz's are left alone.

DECLARE
    TYPE statement_list IS TABLE OF VARCHAR2(512);
    statements statement_list := statement_list();
BEGIN
    -- Collect first: DDL commits, so it never runs inside the dictionary loop.
    FOR col IN (
        SELECT table_name, column_name, data_default
        FROM user_tab_columns
        WHERE data_type LIKE 'TIMESTAMP%'
          AND data_type NOT LIKE '%TIME ZONE'
          AND table_name IN (
              'AGENTS', 'AGENT_LLM_CONFIGS', 'AGENT_MEMORIES',
              'AGENT_KNOWLEDGE_BASES', 'AGENT_DELEGATES', 'PERSONAS', 'TOOLBOX',
              'SKILLBOX', 'CONVERSATION_MEMORY', 'KNOWLEDGE_BASE',
              'SHORT_TERM_MEMORY', 'WORKFLOW_MEMORY', 'SHARED_MEMORY',
              'SUMMARIES', 'SUMMARY_MESSAGE_LINKS', 'SEMANTIC_CACHE',
              'ENTITY_MEMORY', 'TOOL_LOG', 'AUTOMATION_JOBS', 'AUTOMATION_RUNS',
              'AUTOMATION_DELIVERIES'
          )
    ) LOOP
        IF col.data_default IS NOT NULL
           AND REGEXP_LIKE(
               col.data_default,
               '^\s*(CURRENT_TIMESTAMP|LOCALTIMESTAMP|SYSTIMESTAMP)\s*(\(\s*\d*\s*\))?\s*$',
               'i'
           )
        THEN
            statements.EXTEND;
            statements(statements.COUNT) :=
                'ALTER TABLE ' || col.table_name || ' MODIFY (' ||
                col.column_name || ' DEFAULT SYS_EXTRACT_UTC(SYSTIMESTAMP))';
        END IF;
    END LOOP;

    FOR i IN 1 .. statements.COUNT LOOP
        EXECUTE IMMEDIATE statements(i);
    END LOOP;
END;
/
