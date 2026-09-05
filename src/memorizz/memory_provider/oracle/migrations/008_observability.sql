-- Additive private observability index. Run via OracleSpanIndex.initialize()
-- or replace __SCHEMA__ with the validated target schema before SQL*Plus.
-- Source shared-memory bundles and application tables are never modified.
DECLARE
    PROCEDURE ensure_object(ddl VARCHAR2) IS
    BEGIN
        EXECUTE IMMEDIATE ddl;
    EXCEPTION WHEN OTHERS THEN
        IF SQLCODE != -955 THEN RAISE; END IF;
    END;
BEGIN
    ensure_object('CREATE TABLE __SCHEMA__.obs_spans (
        event_key VARCHAR2(64) PRIMARY KEY, bundle_key VARCHAR2(64) NOT NULL,
        user_key VARCHAR2(64) NOT NULL, timestamp VARCHAR2(40) NOT NULL,
        verified NUMBER(1) NOT NULL, payload CLOB CHECK (payload IS JSON),
        application_id VARCHAR2(240), agent_id VARCHAR2(240), memory_id VARCHAR2(240),
        thread_id VARCHAR2(240), root_trace_id VARCHAR2(240), run_id VARCHAR2(240),
        turn_id VARCHAR2(240), kind VARCHAR2(240), status VARCHAR2(240),
        tool_name VARCHAR2(240), job_ref VARCHAR2(240), error_code VARCHAR2(240))');
    ensure_object('CREATE TABLE __SCHEMA__.obs_bundles (
        bundle_key VARCHAR2(64) PRIMARY KEY, timestamp VARCHAR2(40),
        source_record_id VARCHAR2(240), fingerprint VARCHAR2(64),
        payload CLOB CHECK (payload IS JSON))');
    ensure_object('CREATE TABLE __SCHEMA__.obs_resources (
        event_key VARCHAR2(64) NOT NULL, ref_hash VARCHAR2(64) NOT NULL,
        PRIMARY KEY (event_key, ref_hash))');
    ensure_object('CREATE TABLE __SCHEMA__.obs_previews (
        event_key VARCHAR2(64) PRIMARY KEY, timestamp VARCHAR2(40) NOT NULL, preview CLOB)');
    ensure_object('CREATE TABLE __SCHEMA__.obs_state (
        name VARCHAR2(80) PRIMARY KEY, value VARCHAR2(240) NOT NULL)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_agent_thread_time ON __SCHEMA__.obs_spans (agent_id, thread_id, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_root_time ON __SCHEMA__.obs_spans (root_trace_id, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_tenant_time ON __SCHEMA__.obs_spans (user_key, application_id, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_kind_status_time ON __SCHEMA__.obs_spans (kind, status, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_memory_time ON __SCHEMA__.obs_spans (memory_id, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_job_time ON __SCHEMA__.obs_spans (job_ref, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_error_time ON __SCHEMA__.obs_spans (error_code, timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_ref ON __SCHEMA__.obs_resources (ref_hash, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_time ON __SCHEMA__.obs_spans (timestamp, event_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_bundle_key ON __SCHEMA__.obs_spans (bundle_key)');
    ensure_object('CREATE INDEX __SCHEMA__.obs_preview_time ON __SCHEMA__.obs_previews (timestamp)');
    -- Resolve obs_state after the dynamic CREATE TABLE, not at block compilation.
    EXECUTE IMMEDIATE 'MERGE INTO __SCHEMA__.obs_state t
        USING (SELECT ''schema_version'' name, ''1'' value FROM dual) s
        ON (t.name = s.name) WHEN NOT MATCHED THEN INSERT (name, value) VALUES (s.name, s.value)';
END;
/
