# Memorizz Local UI

A web-based interface for exploring and managing your Memorizz memory providers locally.

## Overview

The Memorizz Local UI provides a visual dashboard to:

- **Connect** to different memory providers (Oracle, MongoDB, FileSystem)
- **Browse agents** and view their configurations, personas, and tools
- **Inspect context windows** to see conversation history for each agent
- **Explore memory types** including conversations, workflows, entities, summaries, and more

## Quick Start

### 1. Install Dependencies

```bash
pip install memorizz[ui]
```

Or if installing all optional dependencies:

```bash
pip install memorizz[all]
```

### 2. Start the UI

```bash
memorizz ui
```

The server will start at `http://127.0.0.1:8765`

### 3. Connect to Your Memory Provider

Open your browser and navigate to `http://127.0.0.1:8765`. You'll see a connection page where you can select your memory provider and enter credentials.

## CLI Options

```bash
memorizz ui [OPTIONS]

Options:
  --port PORT    Port to run the server on (default: 8765)
  --host HOST    Host to bind to (default: 127.0.0.1)
```

### Examples

```bash
# Start on default port 8765
memorizz ui

# Start on custom port
memorizz ui --port 9000

# Make accessible from other machines on your network
memorizz ui --host 0.0.0.0 --port 8080
```

## Memory Providers

### Oracle Database

| Field | Description | Required |
|-------|-------------|----------|
| Username | Oracle database username | Yes |
| Password | Oracle database password | Yes |
| DSN | Data Source Name (e.g., `localhost:1521/FREEPDB1`) | Yes |
| Schema | Schema name (defaults to username) | No |

For consistent Oracle embedding behavior across UI and notebooks, set shared defaults in your environment:

```bash
export MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER=openai
export MEMORIZZ_DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
export MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS=1536
```

### MongoDB

| Field | Description | Required |
|-------|-------------|----------|
| Connection URI | MongoDB connection string (e.g., `mongodb://localhost:27017`) | Yes |
| Database Name | Database to use (defaults to `memorizz`) | No |

### FileSystem

| Field | Description | Required |
|-------|-------------|----------|
| Storage Path | Local directory path for data storage | Yes |

## Features

### Dashboard

The main dashboard displays:
- Current connection status and provider info
- Statistics for each memory type (agents, personas, tools, etc.)
- Quick navigation to any memory type

### Agents View

Browse all agents stored in your memory provider:
- Agent cards showing persona name, application mode, and memory count
- Click any agent to open it in the playground

### Agent Playground

Interactive workspace for each agent:
- **Chat**: converse with the agent, streaming responses
- **Context Panel** (right): system instruction, conversation, toolbox memory,
  workflow memory, entity memory, summary memory, knowledge base entries,
  tool logs
- **📎 Attach** / **drag-and-drop**: drop one or more files onto the chat
  area, or click the Attach button. The UI calls
  `KnowledgeBase.ingest_file` per upload — the same SDK method you'd use
  from Python — so every registered format and every registered extractor
  is automatically supported. Each upload is chunked (default: `fixed`,
  1000 chars, 100-char overlap), embedded, stored under a new
  `knowledge_base_id`, and attached to this agent. The Knowledge Base
  panel refreshes when ingest completes.
    - Plain-text (`.txt`, `.md`, `.json`, `.csv`, `.py`, `.html`, …) works
      out of the box.
    - **PDFs** need `pypdf`: `pip install 'memorizz[ingest-pdf]'`.
      Scanned/image-only PDFs produce no extractable text and will return
      an error — they need OCR, which is not bundled.
    - Add other formats via `register_extractor` in the SDK — see
      [`semantic/README.md`](../long_term/semantic/README.md#supported-formats--adding-your-own).

### Memory Type Views

Each memory type has its own list view:
- Personas
- Toolbox (Tools)
- Conversations
- Workflows
- Knowledge Base
- Short-term Memory
- Entity Memory
- Summaries
- Shared Memory
- Semantic Cache

## Architecture

```
src/memorizz/ui/
├── __init__.py          # Package exports
├── app.py               # FastAPI application & routes
├── api/                 # API endpoints (future expansion)
├── templates/           # Jinja2 HTML templates
│   ├── base.html        # Base layout with sidebar
│   ├── connect.html     # Provider connection page
│   ├── dashboard.html   # Main dashboard
│   ├── agents.html      # Agent list
│   ├── agent_form.html  # Create/edit agent form
│   ├── playground.html  # Agent playground (also the agent landing page)
│   └── memory_list.html # Generic memory type list
└── static/
    ├── css/
    │   └── styles.css   # UI styling
    └── js/
        └── app.js       # Minimal JavaScript
```

## API Endpoints

The UI also exposes JSON API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Connection status |
| `/api/agents` | GET | List all agents |
| `/api/agents/{id}` | GET | Get agent details |

## Security Notes

⚠️ **Important**: The local UI is intended for development and local use only.

- The server binds to `127.0.0.1` by default (localhost only)
- No authentication is implemented
- Do not expose to public networks without additional security measures
- Credentials are handled in-memory and not persisted

## Troubleshooting

### Port Already in Use

```bash
# Use a different port
memorizz ui --port 9000
```

### Connection Failed

- Verify your memory provider is running and accessible
- Check credentials and connection strings
- For Oracle: Ensure the Oracle client is installed
- For MongoDB: Verify the URI format and database access

### Missing Dependencies

```bash
# Reinstall with UI dependencies
pip install --upgrade memorizz[ui]
```

## Development

To run during development:

```bash
cd /path/to/memorizz
PYTHONPATH=src python -m memorizz.cli run local
```

Or use uvicorn directly:

```python
from memorizz.ui import create_app
import uvicorn

app = create_app()
uvicorn.run(app, host="127.0.0.1", port=8765, reload=True)
```
