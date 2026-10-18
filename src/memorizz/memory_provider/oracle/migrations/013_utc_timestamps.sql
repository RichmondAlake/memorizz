-- One-off conversion of the TIMESTAMP values MemoRizz wrote before it stored
-- UTC.
--
-- NOT applied automatically, and refuses to run twice: it rewrites data.
-- OracleProvider never runs it; run it by hand (SQL*Plus, SQLcl, SQL
-- Developer) as the schema owner after setting source_tz below.
--
-- Why: older releases wrote plain TIMESTAMP columns from local clocks.
--   * CURRENT_TIMESTAMP (column defaults and store paths) and datetime.now()
--     values (conversation and tool-log timestamps, skill promotion stamps)
--     are wall-clock time in the zone of the machine that ran MemoRizz:
--     python-oracledb thin mode sets each session's time zone to the Python
--     process's UTC offset (or ORA_SDTZ). They are converted from source_tz.
--   * SYSTIMESTAMP (shared memory rows) is the database server's clock; those
--     columns are converted from database_tz (the Oracle container images run
--     in UTC, where nothing changes).
--   * updated_at on personas, toolbox, skillbox, knowledge_base,
--     workflow_memory and entity_memory was written by both clocks (saves
--     used the session clock, update_by_id the server's); it is converted
--     from source_tz, so where the server ran in another zone those values
--     stay off by the difference. Nothing filters or expires on updated_at.
--   * expires_at is already UTC and is left alone, as is
--     knowledge_base.last_accessed (no MemoRizz writer).
--   * automation_jobs/automation_runs schedule times (TIMESTAMP WITH TIME
--     ZONE) were bound without their zone, so each holds the intended UTC
--     wall clock labelled with the session's zone. Those are relabelled as
--     UTC exactly (no parameter needed; values already in UTC are skipped).
-- Thin mode fixes the session offset when a pooled connection opens, so a
-- value written across a daylight-saving change by an older connection may
-- stay an hour off.
--
-- Run it before this release first writes (or with MemoRizz stopped right
-- after the upgrade). If the release has already run, set convert_before.
-- The whole conversion is one transaction: an error leaves the data as it
-- was. 012_utc_timestamp_defaults.sql (applied at startup) makes new rows
-- UTC either way.

DECLARE
    -- REQUIRED. The time zone the old values are in: the zone of the machine
    -- that ran MemoRizz. A region keeps daylight saving right, e.g.
    -- 'Europe/London' or 'America/New_York'; a fixed offset like '+05:30'
    -- also works.
    source_tz CONSTANT VARCHAR2(64) := 'SET_SOURCE_TIME_ZONE';
    -- The database server's time zone (SYSTIMESTAMP); NULL means source_tz.
    database_tz CONSTANT VARCHAR2(64) := NULL;
    -- OPTIONAL. When this release has already written UTC rows, the moment
    -- it first started, as wall-clock time in source_tz: values at or after
    -- it are left alone. NULL converts every value. Zones ahead of UTC leave
    -- rows written within that offset after the upgrade ambiguous.
    convert_before CONSTANT TIMESTAMP := NULL;

    migration_name CONSTANT VARCHAR2(64) := '013_utc_timestamps';
    applied NUMBER;

    PROCEDURE convert_column(
        table_name_in VARCHAR2,
        column_name_in VARCHAR2,
        zone_in VARCHAR2
    ) IS
        column_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO column_count
        FROM user_tab_columns
        WHERE table_name = UPPER(table_name_in)
          AND column_name = UPPER(column_name_in)
          AND data_type LIKE 'TIMESTAMP%'
          AND data_type NOT LIKE '%TIME ZONE';
        IF column_count = 1 THEN
            EXECUTE IMMEDIATE
                'UPDATE ' || table_name_in ||
                ' SET ' || column_name_in ||
                ' = SYS_EXTRACT_UTC(FROM_TZ(' || column_name_in || ', :zone))' ||
                ' WHERE ' || column_name_in || ' IS NOT NULL' ||
                ' AND (:before_a IS NULL OR ' || column_name_in || ' < :before_b)'
                USING zone_in, convert_before, convert_before;
        END IF;
    END;

    PROCEDURE relabel_as_utc(table_name_in VARCHAR2, column_name_in VARCHAR2) IS
        column_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO column_count
        FROM user_tab_columns
        WHERE table_name = UPPER(table_name_in)
          AND column_name = UPPER(column_name_in)
          AND data_type LIKE '%WITH TIME ZONE';
        IF column_count = 1 THEN
            EXECUTE IMMEDIATE
                'UPDATE ' || table_name_in ||
                ' SET ' || column_name_in ||
                ' = FROM_TZ(CAST(' || column_name_in || ' AS TIMESTAMP), ''UTC'')' ||
                ' WHERE ' || column_name_in || ' IS NOT NULL' ||
                ' AND (EXTRACT(TIMEZONE_HOUR FROM ' || column_name_in || ') <> 0' ||
                ' OR EXTRACT(TIMEZONE_MINUTE FROM ' || column_name_in || ') <> 0)';
        END IF;
    END;
BEGIN
    IF source_tz = 'SET_SOURCE_TIME_ZONE' THEN
        RAISE_APPLICATION_ERROR(
            -20001,
            'Set source_tz at the top of 013_utc_timestamps.sql before running it'
        );
    END IF;

    SELECT COUNT(*) INTO applied
    FROM user_tables
    WHERE table_name = 'MEMORIZZ_DATA_MIGRATIONS';
    IF applied = 0 THEN
        EXECUTE IMMEDIATE
            'CREATE TABLE memorizz_data_migrations (' ||
            'name VARCHAR2(128) PRIMARY KEY, ' ||
            'applied_at TIMESTAMP DEFAULT SYS_EXTRACT_UTC(SYSTIMESTAMP), ' ||
            'detail VARCHAR2(4000))';
    END IF;
    EXECUTE IMMEDIATE
        'SELECT COUNT(*) FROM memorizz_data_migrations WHERE name = :name'
        INTO applied USING migration_name;
    IF applied > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            migration_name || ' was already applied; running it again would shift the values twice'
        );
    END IF;

    -- Session clock (CURRENT_TIMESTAMP) and datetime.now() values.
    FOR item IN (
        SELECT 'agents' AS t, 'created_at' AS c FROM dual
        UNION ALL SELECT 'agents', 'updated_at' FROM dual
        UNION ALL SELECT 'agent_memories', 'created_at' FROM dual
        UNION ALL SELECT 'agent_knowledge_bases', 'created_at' FROM dual
        UNION ALL SELECT 'agent_delegates', 'created_at' FROM dual
        UNION ALL SELECT 'personas', 'created_at' FROM dual
        UNION ALL SELECT 'personas', 'updated_at' FROM dual
        UNION ALL SELECT 'toolbox', 'created_at' FROM dual
        UNION ALL SELECT 'toolbox', 'updated_at' FROM dual
        UNION ALL SELECT 'skillbox', 'created_at' FROM dual
        UNION ALL SELECT 'skillbox', 'updated_at' FROM dual
        UNION ALL SELECT 'skillbox', 'promoted_at' FROM dual
        UNION ALL SELECT 'skillbox', 'demoted_at' FROM dual
        UNION ALL SELECT 'conversation_memory', 'timestamp' FROM dual
        UNION ALL SELECT 'conversation_memory', 'last_accessed_at' FROM dual
        UNION ALL SELECT 'knowledge_base', 'created_at' FROM dual
        UNION ALL SELECT 'knowledge_base', 'updated_at' FROM dual
        UNION ALL SELECT 'knowledge_base', 'last_accessed_at' FROM dual
        UNION ALL SELECT 'short_term_memory', 'created_at' FROM dual
        UNION ALL SELECT 'workflow_memory', 'created_at' FROM dual
        UNION ALL SELECT 'workflow_memory', 'updated_at' FROM dual
        UNION ALL SELECT 'workflow_memory', 'last_accessed_at' FROM dual
        UNION ALL SELECT 'summaries', 'created_at' FROM dual
        UNION ALL SELECT 'summaries', 'last_accessed_at' FROM dual
        UNION ALL SELECT 'summary_message_links', 'created_at' FROM dual
        UNION ALL SELECT 'semantic_cache', 'created_at' FROM dual
        UNION ALL SELECT 'entity_memory', 'created_at' FROM dual
        UNION ALL SELECT 'entity_memory', 'updated_at' FROM dual
        UNION ALL SELECT 'entity_memory', 'last_accessed_at' FROM dual
        UNION ALL SELECT 'tool_log', 'timestamp' FROM dual
        UNION ALL SELECT 'automation_jobs', 'created_at' FROM dual
        UNION ALL SELECT 'automation_jobs', 'updated_at' FROM dual
        UNION ALL SELECT 'automation_runs', 'created_at' FROM dual
        UNION ALL SELECT 'automation_deliveries', 'created_at' FROM dual
        UNION ALL SELECT 'automation_deliveries', 'updated_at' FROM dual
    ) LOOP
        convert_column(item.t, item.c, source_tz);
    END LOOP;

    -- Database server clock (SYSTIMESTAMP).
    convert_column('shared_memory', 'created_at', NVL(database_tz, source_tz));
    convert_column('shared_memory', 'updated_at', NVL(database_tz, source_tz));

    -- Schedule times bound without their zone.
    relabel_as_utc('automation_jobs', 'start_at');
    relabel_as_utc('automation_jobs', 'next_run_at');
    relabel_as_utc('automation_jobs', 'last_run_at');
    relabel_as_utc('automation_jobs', 'lock_expires_at');
    relabel_as_utc('automation_runs', 'scheduled_for');
    relabel_as_utc('automation_runs', 'started_at');
    relabel_as_utc('automation_runs', 'finished_at');

    EXECUTE IMMEDIATE
        'INSERT INTO memorizz_data_migrations (name, detail) VALUES (:name, :detail)'
        USING migration_name,
              'source_tz=' || source_tz ||
              ' database_tz=' || NVL(database_tz, source_tz) ||
              ' convert_before=' || NVL(TO_CHAR(convert_before), '-');
    COMMIT;
END;
/
