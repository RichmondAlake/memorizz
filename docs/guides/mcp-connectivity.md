# MCP Connectivity

Streaming is now the default delivery path. See the
[streaming contract, defaults and compatibility modes](streaming.md)
for the SDK event iterator, CLI opt-out, UI lifecycle and opt-in MCP answer events.
Full-answer completion validators still buffer until acceptance; Python `run()`
retains its complete-string return contract.

Memorizz has a first-class Model Context Protocol client built on the official
Python SDK. Agents can discover and call tools, list/read resources, expand
resource templates, and list/get prompts over:

- local `stdio` subprocesses;
- Streamable HTTP (the current remote transport); and
- legacy HTTP+SSE servers.

MCP calls are available to the agent through `mcp_*` facade tools, in the local
UI under **MCP Connections**, and through `memorizz mcp`.

Install the optional protocol and credential dependencies first:

```bash
pip install "memorizz[mcp]"
# For the MCP management UI as well:
pip install "memorizz[mcp,ui]"
```

This page covers MemoRizz as an MCP client. To make MemoRizz memory and agents
available to other MCP clients, see [Expose MemoRizz as an MCP server](mcp-server.md).

## Notion

The easiest UI flow is:

1. Start `memorizz ui` and open **MCP Connections**.
2. Select an agent, choose **Connect Notion**, and save.
3. Select **Authorize**, approve access in Notion, then use **Test**.

The preset uses Notion's hosted endpoint:

```text
https://mcp.notion.com/mcp
```

Notion supports OAuth and integration tokens. For an integration token, select
**Bearer token / PAT** instead of OAuth and paste the token; the secret is
encrypted outside the agent record. See Notion's
[MCP client guide](https://developers.notion.com/guides/mcp/build-mcp-client)
and [connection guide](https://developers.notion.com/guides/mcp/get-started-with-mcp).

CLI OAuth:

```bash
memorizz mcp add notion --preset notion
memorizz mcp login notion
memorizz mcp test notion
memorizz mcp tools notion
```

The terminal login opens a browser and asks you to paste the full final callback
URL. This also works when the browser cannot load the loopback page: copy the URL
from its address bar.

## Google Calendar

Google Calendar MCP is currently a Google Developer Preview. Before connecting,
enable the Calendar API's MCP server and create an OAuth web client in the same
Google Cloud project. Register the exact redirect URI shown by the Memorizz UI,
normally:

```text
http://127.0.0.1:8765/api/mcp/oauth/callback
```

Then choose **Connect Google Calendar**, enter the OAuth client ID and client
secret, save, and authorize. The preset uses:

```text
https://calendarmcp.googleapis.com/mcp/v1
```

CLI setup:

```bash
memorizz mcp add calendar \
  --preset google-calendar \
  --client-id "$GOOGLE_OAUTH_CLIENT_ID" \
  --client-secret "$GOOGLE_OAUTH_CLIENT_SECRET"
memorizz mcp login calendar
memorizz mcp test calendar
```

Use Google's [Calendar MCP setup guide](https://developers.google.com/workspace/calendar/api/guides/configure-mcp-server)
for Cloud project, OAuth consent-screen, and access-policy requirements.

## Local and custom servers

Add a local SDK server without a shell:

```bash
memorizz mcp add local-files \
  --transport stdio \
  --command uvx \
  --arg mcp-server-filesystem \
  --arg /absolute/allowed/path
```

Add a custom remote server:

```bash
memorizz mcp add company \
  --transport streamable_http \
  --url https://mcp.example.com/mcp \
  --auth bearer \
  --token "$COMPANY_MCP_TOKEN"
```

For scripts and deployed applications, pass public configuration when building
an agent. Inline secrets are extracted into the configured credential store:

```python
agent = MemAgent(
    model=model,
    memory_provider=provider,
    mcp_servers=[
        {
            "name": "notion",
            "transport": "streamable_http",
            "url": "https://mcp.notion.com/mcp",
            "auth": {
                "type": "oauth",
                "redirect_uri": "https://app.example.com/api/mcp/oauth/callback",
            },
            "require_approval": True,
        }
    ],
)

agent.with_mcp_servers([...])  # replace connections at runtime
```

## CLI reference

```text
memorizz mcp list [--json]
memorizz mcp add NAME [connection/auth/policy options]
memorizz mcp remove NAME
memorizz mcp status [NAME] [--json]
memorizz mcp login NAME
memorizz mcp logout NAME
memorizz mcp test NAME
memorizz mcp tools|resources|prompts NAME
memorizz mcp call NAME TOOL --arguments '{"key":"value"}'
memorizz mcp approvals [--status pending]
memorizz mcp approve PROPOSAL_ID --approver OPERATOR_ID
memorizz mcp resume PROPOSAL_ID
memorizz mcp reject|cancel PROPOSAL_ID --approver OPERATOR_ID
memorizz mcp serve [--transport stdio|streamable-http]
```

## Interpreting hosted connection tests

Hosted MCP reachability and authorization are separate states:

- `authorization_required` proves that the endpoint was reached and issued an
  authentication challenge; it does **not** mean OAuth succeeded;
- some servers expose discovery before login, so a tool count can be available
  while a protected call still returns `authorization_required`;
- `approval_required` is MemoRizz's local durable mutation gate and is distinct
  from the remote server's OAuth grant;
- a successful production test requires an operator-owned OAuth grant followed
  by an actual permitted read call. Notion/Google credentials cannot be
  manufactured by a package test suite.

For release validation, cover unauthenticated, OAuth-required,
approval-required, and successful paths. Use a dedicated test workspace and
Google Cloud project for the successful path, and never store refresh tokens in
fixtures or agent JSON.

## Security and reliability defaults

- Bearer tokens, OAuth client secrets/tokens, custom headers, and stdio
  environment values are encrypted with Fernet in `~/.memorizz`; the public
  agent configuration stores only names and credential references.
- Set `MEMORIZZ_MCP_ENCRYPTION_KEY` to a stable Fernet key in containers or
  multi-instance deployments. Back up the key separately; tokens cannot be
  recovered without it.
- Remote connections require HTTPS and reject private, loopback, link-local,
  reserved, and rebinding/redirect destinations by default. Use
  `allow_private_network` only for an intentional local service.
- `MEMORIZZ_MCP_HOST_ALLOWLIST` restricts remote destinations with comma-separated
  host patterns. `MEMORIZZ_MCP_STDIO_ALLOWLIST` similarly restricts executable
  names or paths.
- Safe discovery/read operations use bounded exponential retries. Tool calls
  that may mutate external systems are never retried automatically.
- Tool allow/block lists are enforced locally. A mutating call creates a
  durable proposal instead of accepting a model- or caller-controlled Boolean.
  A host operator approves or rejects it with an audited identity and then
  resumes the exact stored call. Proposals expire and are single-use.
- Responses are bounded by `max_result_bytes` (2 MB by default), timeouts are
  configurable, and tool-call audit records contain argument hashes rather than
  argument values.

For hosted deployments, implement the `CredentialStore` protocol with your
cloud secret manager and pass it to `MCPClientManager`. Run OAuth callbacks on a
stable HTTPS origin and register that exact redirect URI with each provider.
