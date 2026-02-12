-- ==============================================================================
-- SIMPLIFIED JSON Relational Duality Views for Memorizz
-- Oracle 23ai+ feature - WORKING VERSION
--
-- Note: Columns with IS JSON constraints cannot be in Duality Views
-- This version excludes: traits, expertise, parameters, steps, outcome,
-- access_list, original_memory_ids, additional_config
-- ==============================================================================

-- ==============================================================================
-- AGENTS DUALITY VIEW (standalone - no nesting for now)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW agents_dv AS
agents @insert @update @delete
{
    _id        : id,
    agentId    : agent_id,
    name       : name,
    instruction: instruction,
    applicationMode: application_mode,
    maxSteps   : max_steps,
    toolAccess : tool_access,
    semanticCache: semantic_cache,
    isFavorite : is_favorite,
    verbose    : verbose,
    embedding  : embedding,
    createdAt  : created_at,
    updatedAt  : updated_at
};

-- ==============================================================================
-- PERSONAS DUALITY VIEW
-- Note: Excludes 'traits' and 'expertise' (IS JSON columns)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW personas_dv AS
personas @insert @update @delete
{
    _id       : id,
    personaId : persona_id,
    name      : name,
    roleType  : role_type,
    background: background,
    memoryId  : memory_id,
    agentId   : agent_id,
    embedding : embedding,
    createdAt : created_at,
    updatedAt : updated_at
};

-- ==============================================================================
-- TOOLBOX DUALITY VIEW
-- Note: Excludes 'parameters' (IS JSON column)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW toolbox_dv AS
toolbox @insert @update @delete
{
    _id       : id,
    toolId    : tool_id,
    name      : name,
    description: description,
    signature : signature,
    docstring : docstring,
    toolType  : tool_type,
    memoryId  : memory_id,
    agentId   : agent_id,
    embedding : embedding,
    createdAt : created_at,
    updatedAt : updated_at
};

-- ==============================================================================
-- CONVERSATION_MEMORY DUALITY VIEW
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW conversation_memory_dv AS
conversation_memory @insert @update @delete
{
    _id          : id,
    memoryId     : memory_id,
    conversationId: conversation_id,
    role         : role,
    content      : content,
    timestamp    : timestamp,
    agentId      : agent_id,
    embedding    : embedding
};

-- ==============================================================================
-- LONG_TERM_MEMORY DUALITY VIEW
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW long_term_memory_dv AS
long_term_memory @insert @update @delete
{
    _id         : id,
    memoryId    : memory_id,
    content     : content,
    memoryType  : memory_type,
    importance  : importance,
    lastAccessed: last_accessed,
    accessCount : access_count,
    agentId     : agent_id,
    embedding   : embedding,
    createdAt   : created_at,
    updatedAt   : updated_at
};

-- ==============================================================================
-- SHORT_TERM_MEMORY DUALITY VIEW
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW short_term_memory_dv AS
short_term_memory @insert @update @delete
{
    _id       : id,
    memoryId  : memory_id,
    content   : content,
    memoryType: memory_type,
    ttl       : ttl,
    agentId   : agent_id,
    embedding : embedding,
    createdAt : created_at,
    expiresAt : expires_at
};

-- ==============================================================================
-- WORKFLOW_MEMORY DUALITY VIEW
-- Note: Excludes 'steps' and 'outcome' (IS JSON columns)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW workflow_memory_dv AS
workflow_memory @insert @update @delete
{
    _id        : id,
    workflowId : workflow_id,
    name       : name,
    description: description,
    currentStep: current_step,
    status     : status,
    memoryId   : memory_id,
    agentId    : agent_id,
    embedding  : embedding,
    createdAt  : created_at,
    updatedAt  : updated_at
};

-- ==============================================================================
-- SHARED_MEMORY DUALITY VIEW
-- Note: Excludes 'access_list' (IS JSON column)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW shared_memory_dv AS
shared_memory @insert @update @delete
{
    _id        : id,
    memoryId   : memory_id,
    content    : content,
    memoryType : memory_type,
    scope      : scope,
    ownerAgentId: owner_agent_id,
    embedding  : embedding,
    createdAt  : created_at,
    updatedAt  : updated_at
};

-- ==============================================================================
-- SUMMARIES DUALITY VIEW
-- Note: Excludes 'original_memory_ids' (IS JSON column)
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW summaries_dv AS
summaries @insert @update @delete
{
    _id        : id,
    summaryId  : summary_id,
    content    : content,
    summaryType: summary_type,
    memoryId   : memory_id,
    agentId    : agent_id,
    embedding  : embedding,
    createdAt  : created_at
};

-- ==============================================================================
-- SEMANTIC_CACHE DUALITY VIEW
-- ==============================================================================
CREATE OR REPLACE JSON RELATIONAL DUALITY VIEW semantic_cache_dv AS
semantic_cache @insert @update @delete
{
    _id               : id,
    cacheKey          : cache_key,
    queryText         : query_text,
    response          : response,
    scope             : scope,
    similarityThreshold: similarity_threshold,
    hitCount          : hit_count,
    agentId           : agent_id,
    embedding         : embedding,
    createdAt         : created_at,
    expiresAt         : expires_at
};

-- ==============================================================================
-- COMMENTS
-- ==============================================================================
COMMENT ON VIEW agents_dv IS 'JSON Duality View for agents';
COMMENT ON VIEW personas_dv IS 'JSON Duality View for personas';
COMMENT ON VIEW toolbox_dv IS 'JSON Duality View for toolbox';
COMMENT ON VIEW conversation_memory_dv IS 'JSON Duality View for conversation memory';
COMMENT ON VIEW long_term_memory_dv IS 'JSON Duality View for long-term memory';
COMMENT ON VIEW short_term_memory_dv IS 'JSON Duality View for short-term memory';
COMMENT ON VIEW workflow_memory_dv IS 'JSON Duality View for workflow memory';
COMMENT ON VIEW shared_memory_dv IS 'JSON Duality View for shared memory';
COMMENT ON VIEW summaries_dv IS 'JSON Duality View for summaries';
COMMENT ON VIEW semantic_cache_dv IS 'JSON Duality View for semantic cache';

-- ==============================================================================
-- USAGE NOTES
-- ==============================================================================
/*
For columns excluded due to IS JSON constraints, access them directly from tables:
- personas.traits, personas.expertise
- toolbox.parameters
- workflow_memory.steps, workflow_memory.outcome
- shared_memory.access_list
- summaries.original_memory_ids
- agent_llm_configs.additional_config

Example:
  SELECT
    p._id,
    p.name,
    base.traits  -- Access JSON column from base table
  FROM personas_dv p
  JOIN personas base ON base.id = HEXTORAW(p._id);
*/
