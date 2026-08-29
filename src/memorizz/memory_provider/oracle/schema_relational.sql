-- ==============================================================================
-- Oracle Relational Schema for Memorizz
-- This replaces the hybrid JSON-in-CLOB approach with proper relational tables
-- ==============================================================================

-- ==============================================================================
-- AGENTS TABLE (Main agent configuration)
-- ==============================================================================
CREATE TABLE agents (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    agent_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255),
    instruction CLOB,
    application_mode VARCHAR2(50) DEFAULT 'assistant',
    max_steps NUMBER(10) DEFAULT 20,
    tool_access VARCHAR2(50) DEFAULT 'private',
    semantic_cache NUMBER(1) DEFAULT 0,  -- Boolean: 0=false, 1=true
    is_favorite NUMBER(1) DEFAULT 0,     -- Boolean: 0=false, 1=true
    verbose NUMBER(1) DEFAULT 0,         -- Boolean: 0=false, 1=true
    embedding VECTOR(256, FLOAT32),       -- For semantic search on agents
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_agents_semantic_cache CHECK (semantic_cache IN (0, 1)),
    CONSTRAINT chk_agents_is_favorite CHECK (is_favorite IN (0, 1)),
    CONSTRAINT chk_agents_verbose CHECK (verbose IN (0, 1))
);

-- Indexes
CREATE INDEX idx_agents_agent_id ON agents(agent_id);
CREATE INDEX idx_agents_application_mode ON agents(application_mode);
CREATE INDEX idx_agents_created_at ON agents(created_at);

-- ==============================================================================
-- AGENT_LLM_CONFIGS TABLE (LLM configuration for agents)
-- ==============================================================================
CREATE TABLE agent_llm_configs (
    agent_id RAW(16) PRIMARY KEY,
    provider VARCHAR2(50) NOT NULL,           -- 'openai', 'azure', 'anthropic', etc.
    model VARCHAR2(100) NOT NULL,             -- 'gpt-4o-mini', 'claude-3', etc.
    temperature NUMBER(3,2),                  -- 0.00 to 1.00
    max_tokens NUMBER(10),
    top_p NUMBER(3,2),
    frequency_penalty NUMBER(3,2),
    presence_penalty NUMBER(3,2),
    additional_config CLOB CHECK (additional_config IS JSON),  -- Provider-specific config

    -- Foreign key
    CONSTRAINT fk_agent_llm_config FOREIGN KEY (agent_id)
        REFERENCES agents(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_agent_llm_provider ON agent_llm_configs(provider);
CREATE INDEX idx_agent_llm_model ON agent_llm_configs(model);

-- ==============================================================================
-- AGENT_MEMORIES TABLE (Many-to-many: agents to memory_ids)
-- ==============================================================================
CREATE TABLE agent_memories (
    agent_id RAW(16) NOT NULL,
    memory_id VARCHAR2(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Composite primary key
    PRIMARY KEY (agent_id, memory_id),

    -- Foreign key
    CONSTRAINT fk_agent_memories FOREIGN KEY (agent_id)
        REFERENCES agents(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_agent_memories_memory_id ON agent_memories(memory_id);

-- ==============================================================================
-- AGENT_KNOWLEDGE_BASES TABLE (Many-to-many: agents to knowledge_base ingests)
-- ==============================================================================
-- Each row links an agent to one knowledge_base_id (an ingest-scoped group of
-- chunks in the `knowledge_base` table). Separate from agent_memories so the
-- two id spaces can't collide and we can filter retrievals cleanly.
CREATE TABLE agent_knowledge_bases (
    agent_id RAW(16) NOT NULL,
    knowledge_base_id VARCHAR2(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (agent_id, knowledge_base_id),

    CONSTRAINT fk_agent_knowledge_bases FOREIGN KEY (agent_id)
        REFERENCES agents(id) ON DELETE CASCADE
);

CREATE INDEX idx_agent_kb_knowledge_base_id ON agent_knowledge_bases(knowledge_base_id);

-- ==============================================================================
-- AGENT_DELEGATES TABLE (Many-to-many: agents to delegate agents)
-- ==============================================================================
CREATE TABLE agent_delegates (
    agent_id RAW(16) NOT NULL,
    delegate_agent_id RAW(16) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Composite primary key
    PRIMARY KEY (agent_id, delegate_agent_id),

    -- Foreign keys
    CONSTRAINT fk_agent_delegates_parent FOREIGN KEY (agent_id)
        REFERENCES agents(id) ON DELETE CASCADE,
    CONSTRAINT fk_agent_delegates_child FOREIGN KEY (delegate_agent_id)
        REFERENCES agents(id) ON DELETE CASCADE
);

-- ==============================================================================
-- PERSONAS TABLE (Agent personas/roles)
-- ==============================================================================
CREATE TABLE personas (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    persona_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255) NOT NULL,
    role_type VARCHAR2(50),                   -- 'expert', 'critic', 'coordinator', etc.
    background CLOB,
    traits CLOB CHECK (traits IS JSON),       -- JSON array of traits
    expertise CLOB CHECK (expertise IS JSON), -- JSON array of expertise areas
    memory_id VARCHAR2(255),
    agent_id VARCHAR2(255),
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key to agents
    CONSTRAINT fk_personas_agent FOREIGN KEY (agent_id)
        REFERENCES agents(agent_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_personas_persona_id ON personas(persona_id);
CREATE INDEX idx_personas_role_type ON personas(role_type);
CREATE INDEX idx_personas_memory_id ON personas(memory_id);
CREATE INDEX idx_personas_agent_id ON personas(agent_id);

-- ==============================================================================
-- TOOLBOX TABLE (Agent tools)
-- ==============================================================================
CREATE TABLE toolbox (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    tool_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255) NOT NULL,
    description CLOB,
    signature VARCHAR2(1000),
    docstring CLOB,
    tool_type VARCHAR2(50),
    parameters CLOB CHECK (parameters IS JSON),  -- legacy properties projection
    input_schema CLOB CHECK (input_schema IS JSON), -- complete JSON Schema
    tool_policy CLOB CHECK (tool_policy IS JSON),
    aliases CLOB CHECK (aliases IS JSON),
    deprecated_arguments CLOB CHECK (deprecated_arguments IS JSON),
    queries CLOB CHECK (queries IS JSON),
    import_reference VARCHAR2(1000),           -- trusted module:qualified_name only
    memory_id VARCHAR2(255),
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key to agents
    CONSTRAINT fk_toolbox_agent FOREIGN KEY (agent_id)
        REFERENCES agents(agent_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_toolbox_tool_id ON toolbox(tool_id);
CREATE INDEX idx_toolbox_name ON toolbox(name);
CREATE INDEX idx_toolbox_tool_type ON toolbox(tool_type);
CREATE INDEX idx_toolbox_memory_id ON toolbox(memory_id);
CREATE INDEX idx_toolbox_agent_id ON toolbox(agent_id);
CREATE INDEX idx_toolbox_user_id ON toolbox(user_id);

-- ==============================================================================
-- SKILLBOX TABLE (Learned skills promoted from workflow trajectories)
-- ==============================================================================
CREATE TABLE skillbox (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    skill_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255) NOT NULL,
    description CLOB,
    content CLOB,                             -- SKILL.md document body
    preconditions CLOB CHECK (preconditions IS JSON),   -- JSON array of applicability conditions
    tools_used CLOB CHECK (tools_used IS JSON),         -- JSON array of tool names
    queries CLOB CHECK (queries IS JSON),               -- JSON array of exemplar queries
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    source_canonical_hash VARCHAR2(255),      -- Trajectory class the skill was distilled from
    source_workflow_ids CLOB CHECK (source_workflow_ids IS JSON),  -- JSON array of workflow IDs
    exemplar_workflow_id VARCHAR2(255),
    status VARCHAR2(50) DEFAULT 'candidate',
    injection_role VARCHAR2(20) DEFAULT 'user' NOT NULL,
    version NUMBER(10) DEFAULT 1,
    promoted_at TIMESTAMP,
    demoted_at TIMESTAMP,
    demotion_reason CLOB,
    baseline CLOB CHECK (baseline IS JSON),   -- JSON object of pre-promotion metrics
    stats CLOB CHECK (stats IS JSON),         -- JSON object of activation stats
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_skillbox_status CHECK (status IN ('candidate', 'shadow', 'active', 'deprecated', 'demoted')),
    CONSTRAINT chk_skillbox_injection_role CHECK (injection_role IN ('user', 'developer')),

    -- Foreign key to agents
    CONSTRAINT fk_skillbox_agent FOREIGN KEY (agent_id)
        REFERENCES agents(agent_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_skillbox_skill_id ON skillbox(skill_id);
CREATE INDEX idx_skillbox_name ON skillbox(name);
CREATE INDEX idx_skillbox_agent_id ON skillbox(agent_id);
CREATE INDEX idx_skillbox_status ON skillbox(status);
CREATE INDEX idx_skillbox_hash ON skillbox(source_canonical_hash);

-- ==============================================================================
-- CONVERSATION_MEMORY TABLE (Conversation history)
-- ==============================================================================
CREATE TABLE conversation_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    memory_id VARCHAR2(255) NOT NULL,
    thread_id VARCHAR2(255),
    role VARCHAR2(50) NOT NULL,              -- 'user', 'assistant', 'system', 'tool'
    content CLOB NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    summary_id VARCHAR2(255),                 -- compaction marker
    embedding VECTOR(256, FLOAT32),

    -- Constraints
    CONSTRAINT chk_conv_role CHECK (role IN ('user', 'assistant', 'system', 'tool'))
);

-- Indexes
CREATE INDEX idx_conv_memory_id ON conversation_memory(memory_id);
CREATE INDEX idx_conv_thread_id ON conversation_memory(thread_id);
CREATE INDEX idx_conv_timestamp ON conversation_memory(timestamp);
CREATE INDEX idx_conv_agent_id ON conversation_memory(agent_id);
CREATE INDEX idx_conv_user_id ON conversation_memory(user_id);
CREATE INDEX idx_conv_memory_user ON conversation_memory(memory_id, user_id);
CREATE INDEX idx_conv_summary_id ON conversation_memory(summary_id);

-- ==============================================================================
-- KNOWLEDGE_BASE TABLE (Facts, knowledge)
-- ==============================================================================
CREATE TABLE knowledge_base (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    memory_id VARCHAR2(255) NOT NULL,
    content CLOB NOT NULL,
    memory_type VARCHAR2(50),
    importance NUMBER(3,2),                   -- 0.00 to 1.00
    last_accessed TIMESTAMP,
    access_count NUMBER(10) DEFAULT 0,
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    -- Chunking metadata: groups every chunk from one ingest under a single id
    -- and records the strategy used so downstream callers can reason about it.
    knowledge_base_id VARCHAR2(64),
    namespace VARCHAR2(255),
    chunk_index NUMBER(10) DEFAULT 0,
    chunk_count NUMBER(10) DEFAULT 1,
    chunking_strategy VARCHAR2(32),
    source_id VARCHAR2(512),                 -- Stable upstream/source provenance
    parent_source_id VARCHAR2(512),          -- Parent chunk or original memory
    linked_source_ids CLOB,                  -- JSON array for multi-source memories
    metadata CLOB,                           -- JSON provenance and event metadata
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_ltm_memory_id ON knowledge_base(memory_id);
CREATE INDEX idx_ltm_memory_type ON knowledge_base(memory_type);
CREATE INDEX idx_ltm_importance ON knowledge_base(importance);
CREATE INDEX idx_ltm_agent_id ON knowledge_base(agent_id);
CREATE INDEX idx_ltm_user_id ON knowledge_base(user_id);
CREATE INDEX idx_kb_knowledge_base_id ON knowledge_base(knowledge_base_id);
CREATE INDEX idx_kb_namespace ON knowledge_base(namespace);

-- ==============================================================================
-- SHORT_TERM_MEMORY TABLE (Working memory, temporary context)
-- ==============================================================================
CREATE TABLE short_term_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    memory_id VARCHAR2(255) NOT NULL,
    content CLOB NOT NULL,
    memory_type VARCHAR2(50),
    ttl NUMBER(10),                           -- Time-to-live in seconds
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_stm_memory_id ON short_term_memory(memory_id);
CREATE INDEX idx_stm_expires_at ON short_term_memory(expires_at);
CREATE INDEX idx_stm_agent_id ON short_term_memory(agent_id);
CREATE INDEX idx_stm_user_id ON short_term_memory(user_id);

-- ==============================================================================
-- WORKFLOW_MEMORY TABLE (Workflow states and outcomes)
-- ==============================================================================
CREATE TABLE workflow_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    workflow_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255) NOT NULL,
    description CLOB,
    steps CLOB CHECK (steps IS JSON),         -- JSON array of workflow steps
    current_step NUMBER(10) DEFAULT 0,
    status VARCHAR2(50) DEFAULT 'pending',
    outcome CLOB CHECK (outcome IS JSON),     -- JSON result of workflow
    memory_id VARCHAR2(255),
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    user_query CLOB,                          -- Original request for diversity gates and distillation
    canonical_hash VARCHAR2(255),             -- Trajectory identity (sha256 of canonical signature)
    canonical_signature CLOB CHECK (canonical_signature IS JSON),  -- JSON array of canonical units
    step_count NUMBER(10),                    -- len(canonical_signature) after retry collapse
    promoted_skill_id VARCHAR2(255),          -- Set when covered by an ACTIVE learned skill
    skills_activated CLOB CHECK (skills_activated IS JSON),  -- JSON array of skill IDs in context
    shadow_evaluations CLOB CHECK (shadow_evaluations IS JSON),  -- Passive SHADOW evidence
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_workflow_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Indexes
CREATE INDEX idx_workflow_workflow_id ON workflow_memory(workflow_id);
CREATE INDEX idx_workflow_status ON workflow_memory(status);
CREATE INDEX idx_workflow_memory_id ON workflow_memory(memory_id);
CREATE INDEX idx_workflow_agent_id ON workflow_memory(agent_id);
CREATE INDEX idx_workflow_user_id ON workflow_memory(user_id);
CREATE INDEX idx_workflow_canonical_hash ON workflow_memory(canonical_hash);
CREATE INDEX idx_workflow_promoted_skill ON workflow_memory(promoted_skill_id);

-- ==============================================================================
-- SHARED_MEMORY TABLE (Multi-agent shared memory)
-- ==============================================================================
CREATE TABLE shared_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    memory_id VARCHAR2(255) NOT NULL,
    content CLOB NOT NULL,
    memory_type VARCHAR2(50),
    scope VARCHAR2(50) DEFAULT 'global',      -- 'global', 'team', 'private'
    owner_agent_id VARCHAR2(255),
    access_list CLOB CHECK (access_list IS JSON),  -- JSON array of agent IDs
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_shared_scope CHECK (scope IN ('global', 'team', 'private'))
);

-- Indexes
CREATE INDEX idx_shared_memory_id ON shared_memory(memory_id);
CREATE INDEX idx_shared_scope ON shared_memory(scope);
CREATE INDEX idx_shared_owner ON shared_memory(owner_agent_id);

-- ==============================================================================
-- SUMMARIES TABLE (Conversation/memory summaries)
-- ==============================================================================
CREATE TABLE summaries (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    summary_id VARCHAR2(255) UNIQUE NOT NULL,
    content CLOB NOT NULL,
    source_message_ids CLOB CHECK (source_message_ids IS JSON),  -- canonical JSON array
    summary_type VARCHAR2(50),
    memory_id VARCHAR2(255),
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    thread_id VARCHAR2(255),                  -- Exact conversation scope
    period_start NUMBER,
    period_end NUMBER,
    memory_units_count NUMBER(10) DEFAULT 0,
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_summaries_summary_id ON summaries(summary_id);
CREATE INDEX idx_summaries_type ON summaries(summary_type);
CREATE INDEX idx_summaries_memory_id ON summaries(memory_id);
CREATE INDEX idx_summaries_agent_id ON summaries(agent_id);
CREATE INDEX idx_summaries_memory_thread ON summaries(memory_id, thread_id);
CREATE INDEX idx_summaries_user_id ON summaries(user_id);

CREATE TABLE summary_message_links (
    summary_id VARCHAR2(255) NOT NULL,
    message_id RAW(16) NOT NULL,
    position NUMBER(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (summary_id, message_id),
    CONSTRAINT fk_summary_link_summary FOREIGN KEY (summary_id)
        REFERENCES summaries(summary_id) ON DELETE CASCADE,
    CONSTRAINT fk_summary_link_message FOREIGN KEY (message_id)
        REFERENCES conversation_memory(id) ON DELETE CASCADE
);

CREATE INDEX idx_summary_links_message ON summary_message_links(message_id);

-- ==============================================================================
-- SEMANTIC_CACHE TABLE (Query response cache)
-- ==============================================================================
CREATE TABLE semantic_cache (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    cache_key VARCHAR2(255) NOT NULL,
    query_text CLOB NOT NULL,
    response CLOB NOT NULL,
    scope VARCHAR2(50) DEFAULT 'local',       -- 'local', 'global', 'agent'
    similarity_threshold NUMBER(3,2) DEFAULT 0.85,
    hit_count NUMBER(10) DEFAULT 0,
    agent_id VARCHAR2(255),
    memory_id VARCHAR2(255),
    session_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    metadata CLOB CONSTRAINT chk_cache_metadata_json CHECK (metadata IS JSON),
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_cache_scope CHECK (scope IN ('local', 'global', 'agent'))
);

-- Indexes
CREATE INDEX idx_cache_cache_key ON semantic_cache(cache_key);
CREATE INDEX idx_cache_scope ON semantic_cache(scope);
CREATE INDEX idx_cache_agent_id ON semantic_cache(agent_id);
CREATE INDEX idx_cache_user_id ON semantic_cache(user_id);
CREATE INDEX idx_cache_expires_at ON semantic_cache(expires_at);

-- ==============================================================================
-- ENTITY_MEMORY TABLE (Structured entity facts)
-- ==============================================================================
CREATE TABLE entity_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    entity_id VARCHAR2(255) UNIQUE NOT NULL,
    name VARCHAR2(255),
    entity_type VARCHAR2(255),
    attributes CLOB,
    relations CLOB,
    metadata CLOB,
    memory_id VARCHAR2(255),
    agent_id VARCHAR2(255),
    user_id VARCHAR2(255),                    -- Multi-tenant scope (NULL = legacy/anonymous)
    embedding VECTOR(256, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_entity_memory_entity_id ON entity_memory(entity_id);
CREATE INDEX idx_entity_memory_memory_id ON entity_memory(memory_id);
CREATE INDEX idx_entity_memory_agent_id ON entity_memory(agent_id);
CREATE INDEX idx_entity_memory_user_id ON entity_memory(user_id);

-- ==============================================================================
-- TOOL_LOG TABLE (Tool execution logs for context window offloading)
-- ==============================================================================
CREATE TABLE tool_log (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    tool_log_id VARCHAR2(255) UNIQUE NOT NULL,
    tool_name VARCHAR2(255) NOT NULL,
    arguments CLOB,
    result CLOB,
    success NUMBER(1) DEFAULT 1,
    error CLOB,
    outcome VARCHAR2(32) DEFAULT 'success',
    outcome_details CLOB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR2(255),
    tool_call_id VARCHAR2(255),
    thread_id VARCHAR2(255),
    memory_id VARCHAR2(255),
    user_id VARCHAR2(255)                      -- Multi-tenant scope (NULL = legacy/anonymous)
);

-- Indexes
CREATE INDEX idx_tool_log_tool_log_id ON tool_log(tool_log_id);
CREATE INDEX idx_tool_log_memory_id ON tool_log(memory_id);
CREATE INDEX idx_tool_log_agent_id ON tool_log(agent_id);
CREATE INDEX idx_tool_log_thread_id ON tool_log(thread_id);
CREATE INDEX idx_tool_log_tool_name ON tool_log(tool_name);
CREATE INDEX idx_tool_log_user_id ON tool_log(user_id);

-- ==============================================================================
-- AUTOMATIONS TABLES (Durable scheduling + run history)
-- ==============================================================================
CREATE TABLE automation_jobs (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    job_id VARCHAR2(255) UNIQUE NOT NULL,
    agent_id VARCHAR2(255) NOT NULL,
    name VARCHAR2(255) NOT NULL,
    enabled NUMBER(1) DEFAULT 1,
    schedule_type VARCHAR2(20) NOT NULL,     -- 'cron', 'interval', 'one_shot'
    cron_expr VARCHAR2(255),
    interval_seconds NUMBER(10),
    timezone VARCHAR2(64) NOT NULL,
    start_at TIMESTAMP WITH TIME ZONE,
    next_run_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_run_at TIMESTAMP WITH TIME ZONE,
    misfire_policy VARCHAR2(20) DEFAULT 'skip',
    max_run_seconds NUMBER(10) DEFAULT 900,
    retry_max_attempts NUMBER(3) DEFAULT 1,
    retry_backoff_seconds NUMBER(10) DEFAULT 60,
    action_type VARCHAR2(32) NOT NULL,
    action_config CLOB CHECK (action_config IS JSON),
    delivery_type VARCHAR2(32),
    delivery_config CLOB CHECK (delivery_config IS JSON),
    locked_by VARCHAR2(255),
    lock_expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_automation_jobs_enabled CHECK (enabled IN (0, 1))
);

CREATE INDEX idx_automation_jobs_due ON automation_jobs(enabled, next_run_at);
CREATE INDEX idx_automation_jobs_agent ON automation_jobs(agent_id);

CREATE TABLE automation_runs (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    run_id VARCHAR2(255) UNIQUE NOT NULL,
    job_id VARCHAR2(255) NOT NULL,
    scheduled_for TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR2(32) NOT NULL,            -- 'running', 'succeeded', 'failed', 'canceled'
    attempt NUMBER(3) DEFAULT 1,
    error CLOB,
    result_summary CLOB,
    result_payload CLOB CHECK (result_payload IS JSON),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_automation_runs_job ON automation_runs(job_id, created_at);

CREATE TABLE automation_deliveries (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    delivery_id VARCHAR2(255) UNIQUE NOT NULL,
    run_id VARCHAR2(255) NOT NULL,
    channel VARCHAR2(32) NOT NULL,           -- 'whatsapp'
    provider VARCHAR2(32) NOT NULL,          -- 'twilio'
    recipient VARCHAR2(255) NOT NULL,
    status VARCHAR2(32) NOT NULL,            -- 'queued', 'sent', 'failed'
    provider_message_id VARCHAR2(255),
    error CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_automation_deliveries_run ON automation_deliveries(run_id);

-- ==============================================================================
-- VECTOR INDEXES (For similarity search - Oracle 23ai+)
-- ==============================================================================

-- Agents vector index
CREATE VECTOR INDEX idx_agents_vec ON agents(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Personas vector index
CREATE VECTOR INDEX idx_personas_vec ON personas(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Toolbox vector index
CREATE VECTOR INDEX idx_toolbox_vec ON toolbox(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Skillbox vector index
CREATE VECTOR INDEX idx_skillbox_vec ON skillbox(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Conversation memory vector index
CREATE VECTOR INDEX idx_conv_vec ON conversation_memory(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Knowledge base vector index
CREATE VECTOR INDEX idx_kb_vec ON knowledge_base(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Short-term memory vector index
CREATE VECTOR INDEX idx_stm_vec ON short_term_memory(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Workflow memory vector index
CREATE VECTOR INDEX idx_workflow_vec ON workflow_memory(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Shared memory vector index
CREATE VECTOR INDEX idx_shared_vec ON shared_memory(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Summaries vector index
CREATE VECTOR INDEX idx_summaries_vec ON summaries(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Semantic cache vector index
CREATE VECTOR INDEX idx_cache_vec ON semantic_cache(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- Entity memory vector index
CREATE VECTOR INDEX idx_entity_memory_vec ON entity_memory(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
DISTANCE COSINE
WITH TARGET ACCURACY 95;

-- ==============================================================================
-- COMMENTS (Documentation)
-- ==============================================================================

COMMENT ON TABLE agents IS 'Main agents table with configuration';
COMMENT ON TABLE agent_llm_configs IS 'LLM configuration for each agent';
COMMENT ON TABLE agent_memories IS 'Association between agents and memory IDs';
COMMENT ON TABLE agent_delegates IS 'Multi-agent delegation relationships';
COMMENT ON TABLE personas IS 'Agent personas and role configurations';
COMMENT ON TABLE toolbox IS 'Agent tools and functions';
COMMENT ON TABLE skillbox IS 'Learned skills promoted from workflow trajectories';
COMMENT ON TABLE conversation_memory IS 'Conversation history and interactions';
COMMENT ON TABLE knowledge_base IS 'Persistent facts and knowledge';
COMMENT ON TABLE short_term_memory IS 'Temporary working memory with TTL';
COMMENT ON TABLE workflow_memory IS 'Workflow states and execution history';
COMMENT ON TABLE shared_memory IS 'Multi-agent shared memory space';
COMMENT ON TABLE summaries IS 'Memory summaries for compression';
COMMENT ON TABLE semantic_cache IS 'Query-response semantic cache';
COMMENT ON TABLE entity_memory IS 'Structured entity facts and profiles';
COMMENT ON TABLE tool_log IS 'Tool execution logs for context window offloading';
COMMENT ON TABLE automation_jobs IS 'Durable scheduled automation jobs';
COMMENT ON TABLE automation_runs IS 'Execution history for automation jobs';
COMMENT ON TABLE automation_deliveries IS 'Delivery attempts for automation runs';
