# Skills Marketplace

MemoRizz agents can discover and use external agent skills at runtime through pluggable marketplace providers. When a marketplace is enabled, the agent receives search and fetch tools that let it find relevant skills, read their instructions, and apply them to complete tasks.

## Available Providers

| Provider | Identifier | API Key Required | Tools Registered |
| --- | --- | --- | --- |
| **Vercel Agent Skills** | `vercel` | No (search uses skills.sh; optional `GITHUB_TOKEN` raises GitHub rate limits) | `vercel_skills_search`, `vercel_skill_fetch` |
| **SkillsMP** | `skillsmp` | Yes (`SKILLSMP_API_KEY`) | `skills_marketplace_search` |

## Vercel Agent Skills

[Vercel Agent Skills](https://skills.sh) is an open ecosystem of reusable capabilities packaged as `SKILL.md` files in GitHub repositories. Each skill contains structured instructions (with YAML frontmatter for metadata and Markdown content for steps) that an agent can follow to complete a specific type of task.

The ecosystem covers skills for React, Next.js, AI SDK, deployment, design, browser automation, commerce, and more, with 269+ published skills from Vercel and the community.

### How It Works

1. The agent receives `vercel_skills_search` and `vercel_skill_fetch` tools
2. When given a task, the agent searches for relevant skills by keyword
3. The agent fetches the `SKILL.md` from the matching repository
4. The agent reads and follows the instructions to complete the task

### Enabling via Code

```python
from memorizz.memagent import MemAgent

agent = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    skills_marketplace_provider="vercel",
)

print(agent.run("Build a performant React component following best practices"))
```

Or with the builder:

```python
from memorizz.memagent.builders import MemAgentBuilder

agent = (
    MemAgentBuilder()
    .with_instruction("You are a frontend developer.")
    .with_memory_provider(provider)
    .with_llm_config(llm_config)
    .build()
)

# Enable marketplace after build
agent.with_skills_marketplace_provider("vercel")
```

### Enabling via Web UI

1. Navigate to the **Vercel Skills** page in the sidebar to browse and search skills
2. When creating or editing an agent, select **Vercel Agent Skills (skills.sh)** from the **Skills Marketplace** dropdown
3. The agent's playground will show the Vercel provider status in the configuration panel

### Fetching from a Specific Repository

Users can point the agent at any GitHub repository containing a `SKILL.md`:

```python
# The agent fetches the SKILL.md and follows its instructions
agent.run("Use the skill from vercel/ai-chatbot to scaffold a chatbot")
```

The `vercel_skill_fetch` tool accepts:

- `repo` – GitHub repository in `owner/repo` format or a full URL
- `skill_name` – optional name for multi-skill repositories (e.g., `"nextjs-app-router"`)
- `branch` – Git branch to fetch from (default: `main`)

### Multi-Skill Repositories

Some repositories contain multiple skills under a `skills/` directory. When the agent fetches from such a repo without specifying a `skill_name`, it receives a list of available skills and can select the appropriate one.

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `GITHUB_TOKEN` | Optional GitHub personal access token for higher API rate limits |
| `MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER` | Set to `vercel` to auto-enable on all agents |

### How Skills Are Discovered in Repositories

The provider searches these standard paths in order:

- `SKILL.md` (root)
- `skills/SKILL.md`
- `skills/.curated/*/SKILL.md`
- `skills/.experimental/*/SKILL.md`
- `.claude/skills/*/SKILL.md`
- `.agents/skills/*/SKILL.md`
- `.cursor/skills/*/SKILL.md`

This aligns with the [Vercel skills CLI specification](https://github.com/vercel-labs/skills).

### Tool Reference

#### `vercel_skills_search`

Search the Vercel skills ecosystem for relevant skills.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `q` | `str` | required | Search query (e.g., `"react"`, `"deployment"`, `"AI SDK"`) |
| `limit` | `int` | `20` | Maximum results (1–100) |

Returns a dict with `ok`, `skills` (list of repo/name/description/stars), `count`, and `total_count`.

#### `vercel_skill_fetch`

Fetch a skill's instructions from a GitHub repository.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `repo` | `str` | required | Repository in `owner/repo` format or GitHub URL |
| `skill_name` | `str` | `""` | Target a specific skill in a multi-skill repo |
| `branch` | `str` | `"main"` | Branch to fetch from |

Returns a dict with `ok`, `name`, `description`, `instructions` (full SKILL.md content), `repo`, `path`, and `metadata`.

## SkillsMP

The [SkillsMP](https://skillsmp.com) provider connects to a hosted skills marketplace with keyword and AI-powered search.

### Configuration

1. Obtain an API key from [skillsmp.com](https://skillsmp.com)
2. Set it via the **Settings** page in the web UI or export `SKILLSMP_API_KEY`

```python
agent = MemAgent(
    llm_config=llm_config,
    memory_provider=provider,
    skills_marketplace_provider="skillsmp",
    skills_marketplace_config={"api_key": "your-key"},
)
```

### Tool Reference

#### `skills_marketplace_search`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `q` | `str` | required | Search query |
| `mode` | `str` | `"keyword"` | `"keyword"` or `"ai"` for semantic search |
| `page` | `int` | `1` | Page number |
| `limit` | `int` | `20` | Results per page (max 100) |
| `sort_by` | `str` | `""` | `"stars"` or `"recent"` |

## Web UI: Skills page

The local web UI has a **Skills** page in the sidebar. It searches the
[skills.sh](https://skills.sh) directory, ranked by installs, with no token
needed. Paste an `owner/repo` or GitHub URL to fetch a skill directly, and
read the full `SKILL.md` before using it.

### Attach a skill to an agent

Select a result and choose **Attach skill**. MemoRizz saves the skill's
`SKILL.md` under `~/.memorizz/skills/<owner>/<repo>/<skill>/`, records where it
came from in `.memorizz-source.json` (repository, path and SHA-256), and adds
the file to the agent's `skill_paths`. The agent then lists it with
`list_skills` and reads it with `read_skill`. Attaching runs nothing; skill code
only runs in a sandbox through the skill tools.

The same search and **Attach** button are in the playground under
**Settings → Skills**. To detach a skill, remove its row there and save.

Skills from collections and plugins are found wherever they live in the
repository (for example `skills/notion/knowledge-capture/SKILL.md`); when a
name matches nothing, the error lists the skills the repository does contain.

The HTTP API behind these buttons:

```text
GET    /api/agents/{agent_id}/skills
POST   /api/agents/{agent_id}/skills        {"repo": "owner/repo", "skill_name": "name"}
DELETE /api/agents/{agent_id}/skills?path=/absolute/path/SKILL.md
```

Attaching pins a reviewed skill to one agent. Enabling the marketplace instead
lets the agent search and fetch skills on its own at run time.
