# Grok Bot and ChatGPT dots

Grok Bot (xAI), ChatGPT with its dots, and Claude can use your MemoRizz memory
as a custom connector. Every app you approve uses the same memory: something Grok
saves, ChatGPT can find, and the other way round.

Both apps run in the cloud, so they reach MemoRizz only over public HTTPS, and
they sign in with OAuth. `memorizz mcp apps serve` gives you both: it puts the
MCP server online with an OAuth sign-in, and you approve each app with a
one-time pairing code.

## Put the memory online

You need a public HTTPS address that forwards to the server on this computer.
The server itself only listens on `127.0.0.1`, so nothing else on your network
can reach it. A tunnel carries requests from the address to it. Anyone can
open the address, but only an app you approve with a pairing code can use
the memory.

**A fixed address (recommended).** [Tailscale Funnel](https://tailscale.com/kb/1223/funnel)
gives this computer an address like `https://your-mac.your-tailnet.ts.net`
that stays the same, so apps stay connected across restarts. It's free for
personal use:

1. Install Tailscale from the Mac App Store or
   [tailscale.com/download](https://tailscale.com/download), and sign in.
2. Run `tailscale funnel --bg 8766`. The first time, it prints a link to turn
   on HTTPS and Funnel for your tailnet; open it, approve, and run the
   command again. It prints your address.
3. Start the server with that address:

```bash
memorizz mcp apps serve --public-url https://your-mac.your-tailnet.ts.net
```

`--bg` keeps the funnel on after restarts; `tailscale funnel reset` turns it
off. An [ngrok](https://ngrok.com/) domain or a named Cloudflare tunnel on your
own domain works the same way: `ngrok http 8766` prints the address to use. The address must be a bare
origin, without a path.

**A temporary address.** With
[cloudflared](https://github.com/cloudflare/cloudflared/releases) installed:

```bash
memorizz mcp apps serve --tunnel
```

A quick tunnel's address changes every time it starts, so the apps have to be
connected again after a restart.

The command prints the server URL (it ends in `/mcp`) and a pairing code:

```text
Your MemoRizz memory is online for your apps as richmond
  Server URL    https://your-mac.your-tailnet.ts.net/mcp
  Pairing code  KQ7M-4TZP  (one app, 10 minutes; `memorizz mcp apps pair` makes another)
```

Keep it running. Ctrl-C takes the memory offline.

## Connect Grok Bot

The Grok Bot desktop app on your Mac doesn't run connectors itself. Your Bot
works on its own computer in the cloud, and that computer calls the server URL,
which is why the address has to be public.

1. In Grok Bot, open **Connect apps** in the sidebar and choose **Add a custom
   connector**.
2. Name it `memorizz`, keep the type **Remote HTTPS**, and paste the server
   URL. Leave the OAuth client ID and secret empty; Grok Bot signs itself in.
3. Grok Bot opens MemoRizz's approval page in your browser. Type the pairing
   code and choose **Allow**.

You can also ask your Bot in chat: "Add a custom MCP server named memorizz at
https://…/mcp". It confirms the details, then the same approval page opens.

Grok Bot signs in with Cursor's client identity, since Cursor makes the app.
MemoRizz recognises it from where the sign-in returns to, so the approval page
and the apps list call it Grok Bot.

## Connect Grok

1. Go to [grok.com/connectors](https://grok.com/connectors), choose **New
   Connector**, then **Custom**.
2. Enter the server URL.
3. On MemoRizz's approval page, type a new pairing code and choose **Allow**.

Grok finds the memory tools by itself and uses them in your conversations.
It refuses `localhost` and private addresses, and calls your server whenever a
tool is used, so the tunnel and the server have to stay up. MemoRizz lists it
as Grok.

## Connect ChatGPT and your dots

1. In ChatGPT, turn on Developer mode, then go to **Plugins → + → Add custom
   MCP server**. Paste the server URL and choose OAuth.
2. On MemoRizz's approval page, type a new pairing code (`memorizz mcp apps
   pair`) and choose **Allow**.
3. Your dots use the plugins you have in ChatGPT, so there's no separate step.

OpenAI decides what custom servers can do on each plan. On some plans a
custom server can only be read from, so ChatGPT can search your memory
but not save to it.

## Connect Claude

On claude.ai, and in Claude's desktop and mobile apps, go to **Customize →
Connectors → Add custom connector** and paste the server URL. On MemoRizz's
approval page, type a new pairing code and choose **Allow**.

For Claude Code on a computer that can't reach this store directly:

```bash
claude mcp add --transport http memorizz https://your-mac.your-tailnet.ts.net/mcp
claude mcp login memorizz
```

On the computer that holds the store, use the
[Claude Code plugin](coding-agent-plugins.md) instead. It reads the store
directly and saves each session.

## Share your local agents' memory

By default an app has a memory of its own, apart from what your local agents
keep. To give it the same memory as Claude Code, Codex and your other agents on
this computer, make its pairing code with `--shared`, or tick **Share with my
local agents** in the UI:

```bash
memorizz mcp apps pair --shared
memorizz mcp apps serve --shared --public-url https://your-mac.your-tailnet.ts.net
```

No memory is moved. The app acts as your local agents do, so:

- **It finds what your agents know.** It can read their knowledge, entities,
  summaries and conversations, in every project, including past sessions.
- **Your agents find what it saves.** A memory it saves goes to the memory ID
  `personal`, or to the one the user names. New Claude Code and Codex sessions
  list the newest of those at the start, and your agents can search
  `personal` for more.
- **It can change your agents' memories**, and each change records the app.
  Pair with `--shared --read-only` to let an app look but not touch.

A shared app can't use your agents' tool logs, caches or run records. You choose
this for each app, and `memorizz mcp apps list` shows which ones share. Sharing
needs a server that lists only the memory tools, which `memorizz mcp apps serve`
always does. If you set a user with `memorizz plugin install --user NAME`, a
shared app uses that user's memory.

## Pairing codes

The approval page asks for a pairing code so that only you can approve an app,
even though the page is on the public internet. A code:

- approves one app, once;
- expires after 10 minutes (`--minutes` up to 60);
- names whose memory the app uses (`--principal`, by default your user name
  on this computer);
- can share your local agents' memory (`--shared`);
- can make the app read only (`--read-only`).

Make one with `memorizz mcp apps pair` or in the MemoRizz UI under
**Settings → Connected apps**. After five wrong codes the sign-in is refused
and the app has to start again.

## See and disconnect apps

```bash
memorizz mcp apps list
memorizz mcp apps revoke app_3f2a9c1d7e5b4a60
```

Disconnecting takes effect immediately: the app's tokens stop working, and it
has to be approved again to come back. The UI's **Settings → Connected apps**
does the same.

## What the apps can do

Approved apps get the memory tools only:

- `search` and `fetch`, the two read-only tools ChatGPT's deep research and
  company knowledge use;
- search, list and read memories, entities and conversations;
- save and update memories (unless the code was read only). Each saved memory
  records which app saved it (`created_via`), and its change history names
  the app;
- `memorizz_forget_memory`, which only proposes a deletion. It happens
  after you approve it with `memorizz mcp server-approve`.

They can't run agents, harnesses or training. Unless you
[share it](#share-your-local-agents-memory), they don't see the memory your
local agents keep, such as the Claude Code and Codex plugins' project memory.

## How the sign-in works

The server is an OAuth 2.1 authorization server for its own MCP endpoint:

- it publishes protected resource metadata at
  `/.well-known/oauth-protected-resource/mcp` and authorization server metadata
  at `/.well-known/oauth-authorization-server`;
- apps identify themselves with a client metadata document (an HTTPS
  `client_id`, which is how ChatGPT and Grok sign in) or with dynamic client
  registration;
- ChatGPT signs its token requests with its key (`private_key_jwt`), which
  MemoRizz checks against the keys published on ChatGPT's own site;
- a local redirect such as Claude Code's may use any port, since native apps
  listen on whichever one is free (RFC 8252); everything else about a
  redirect has to match exactly;
- PKCE (S256) is required, tokens are issued only for this server's `/mcp`
  endpoint, access tokens last an hour, and refresh tokens are replaced each
  time they're used;
- pairing codes, authorization codes and tokens are stored only as hashes in
  `~/.memorizz/mcp_apps.sqlite3`.

To run the server some other way, `memorizz mcp serve --transport
streamable-http --oauth --tools memory --allow-writes --public-url …` does the
same as `memorizz mcp apps serve`. See
[Expose MemoRizz as an MCP Server](mcp-server.md) for the full set of options.
