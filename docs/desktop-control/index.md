# Desktop Control (Computer Use)

`memorizz desktop` lets an agent open, read and operate the apps on a Mac:
Calculator, Notes, Finder, Mail, System Settings, or any app that exposes its
window to macOS Accessibility. It is an MCP server that MemoRizz starts for the
agent, so the agent gets one tool per action (`desktop__read_window`,
`desktop__click`, …).

The agent reads apps as text, not pixels. Each window becomes a numbered
outline of its buttons, fields and text, and the agent acts by number. No
screenshot is sent to the model, so desktop control works with any
tool-calling model, local ones included.

Actions run in the background. Buttons are pressed through Accessibility, and
keys, text and scrolling are sent to the app's own process, so you can keep
working in another app while the agent drives one. Your typing never lands in
the agent's app, and the agent's typing never lands in yours.

!!! warning "The agent acts with your permissions"
    Desktop control operates real apps as you. Reading a window never asks.
    Opening an app or a link, clicking, typing, pressing keys and scrolling
    each wait for your approval unless you connect with `--no-approval`. Turn
    approval off only for apps and tasks you trust.

## Requirements

- macOS.
- The `desktop` extra: the MCP runtime and the PyObjC bridge to macOS
  Accessibility.
- **Accessibility** permission for the app that starts MemoRizz (Terminal,
  iTerm, VS Code, …). Without it, agents can open apps but not read or operate
  them.
- **Screen Recording** permission, optional: with it, each step keeps a
  screenshot of the app's window.

```bash
pip install "memorizz[desktop]"
# Or, for the CLI installed as a uv tool:
uv tool install --python 3.12 "memorizz[desktop]"

memorizz desktop doctor          # what this Mac allows, and for which app
memorizz desktop doctor --ask    # show macOS's prompts for whatever is missing
```

`doctor` names the app macOS asks about: the `.app` the current terminal runs
in. Turn it on under **System Settings → Privacy & Security → Accessibility**
(and **Screen Recording** for screenshots), then run `doctor` again. Restart
that app if it still says no.

## Turn it on in the chat

In a `memorizz` chat, one command attaches the desktop tools to the agent you
are talking to. No restart is needed, and the agent keeps them next time:

```text
/desktop on                      # computer use for this agent (/computer-use works too)
/desktop                         # on or off, and what macOS allows
/desktop doctor --ask            # show macOS's prompts for whatever is missing
/desktop off                     # remove the tools again
```

`/desktop on --no-screenshots` keeps runs without window screenshots. Each
action asks in the chat before it runs (see [Approvals](#approvals)).

## Connect it from the shell

```bash
memorizz desktop connect         # saves an MCP connection named "desktop"
memorizz desktop doctor
memorizz                         # restart the chat to load the tools
```

`connect` saves a local `stdio` MCP server that runs
`python -m memorizz desktop serve` with the same Python as the CLI, and lists
its tools once so the agent sees each action as a tool of its own.

| Option | Default | Effect |
|---|---|---|
| `--name NAME` | `desktop` | The connection's name; tools are called `NAME__tool`. |
| `--approval / --no-approval` | approval on | Ask before each app launch, click, keystroke or scroll. Reading never asks. |
| `--screenshots / --no-screenshots` | on | Keep a screenshot of the window after each step, with the run's workflow memory. |

The connection is an ordinary MCP server, so `memorizz mcp list`,
`memorizz mcp test desktop` and `memorizz mcp tools desktop` work on it, and it
appears in the UI under **MCP Connections**.

Then ask for something in plain words:

```text
> Open Calculator and work out 12 × 7.
> What does the note called "Groceries" in Notes say?
> In System Settings, which Wi-Fi network am I on?
> Open my "THE ALGORITHM" Notion page in the Notion app.
```

## The tools

| Tool | What it does | Approval |
|---|---|---|
| `list_apps()` | The apps running on this Mac, and which one is in front. | Never |
| `read_window(app, include_menus=False, limit=150)` | A running app's window as numbered elements. `app` defaults to the app used last; `include_menus` adds the menu bar and any open menu. | Never |
| `open_app(name)` | Launch an app that is not running (or switch to it), then read its window. | Yes |
| `open_url(url, app=None)` | Open a link or deep link in an app, then read its window. `app="Notion"` opens a `notion.so` or `app.notion.com` page in the Notion app; without `app` the link opens where macOS usually opens it. `notion://`, `slack://` and `mailto:` links work too; `file:` links are refused. The app comes to the front. | Yes |
| `click(element, double=False)` | Click a numbered element, then read the window again. | Yes |
| `type_text(text, element=None, submit=False, in_front=False)` | Type into a numbered field, or wherever the cursor is; `submit` presses Return. | Yes |
| `press_keys(keys, app=None, in_front=False)` | Keys and shortcuts: `"cmd+n"`, `"down down return"`, `"escape"`. | Yes |
| `scroll(direction="down", amount=5, element=None)` | Scroll the window, or one numbered element, up or down. | Yes |

The server also tells the model to read before acting, and to ask you before
sending, deleting, buying or posting anything.

## What the agent reads

`read_window("Calculator")` returns:

```text
Calculator — window "Calculator"
[1] button "Clear"
[2] button "7"
[3] text field "Search" = "hello" (focused)

Act by number: click(n), type_text(text, element=n), press_keys("cmd+f"), scroll("down"). Numbers change after every read or action.
```

Each line is one element the agent can act on, or a piece of text: buttons,
fields with their current value, checkboxes, tabs, rows, links, menu items and
headings, marked `focused`, `selected` or `disabled` when they are. Elements
scrolled out of view are left out. A window shows its first 150 elements; the
agent can scroll, or read with a higher `limit` (up to 400).

Every action returns the window again with new numbers, so the agent always
acts on what is on screen now. MemoRizz starts a new server process for each
tool call, so the last read is kept in `~/.memorizz/desktop/last-read.json`.
Before acting, the server finds the element again by its place in the window,
its role and its label. If the window has changed since the read, the action
does not run and the agent is told to read again.

## How actions reach the app

- **Clicks** press the element through Accessibility (`AXPress`), focus a
  field, or select a row, so they reach a window behind others. Only an element
  that cannot be pressed falls back to a real pointer click, which brings its
  app to the front.
- **Typing, keys and scrolling** are posted to the app's process. They reach
  the app even when another app is in front. Set `in_front=True` only for an
  app that ignores background input.

## Approvals

Reading is free. Every other tool call becomes a durable approval proposal
that names the exact tool and arguments, and nothing runs until you decide. In
the chat, the turn stops and asks:

```text
Allow desktop__open_url(url="https://app.notion.com/p/8e5d…", app="Notion")? [y/N] y
Opened THE ALGORITHM in your Notion desktop app.
```

Yes runs that exact call and lets the agent carry on; when it reaches the next
action it asks again. Anything else declines it. Ctrl-C leaves the proposal
pending, to decide later with the `/approvals` commands:

```text
/approvals pending
/approvals approve PROPOSAL_ID you@example.com looks right
/approvals resume PROPOSAL_ID
```

In the UI, the Playground shows the same proposal as an approval card with
the exact arguments. See [Tools, Safety & Approvals](../guides/tools-and-approvals.md)
for the full lifecycle.

## Every run is kept as workflow memory

Each desktop task is stored as a [workflow memory](../memory-types/procedural.md)
record, the procedural memory of what the agent did: the request, each tool
call with its arguments and result (the window it read), and the outcome. An
approval pause keeps the same record, so one task is one record however many
times it waits for you.

With Screen Recording allowed, each read and action also saves a JPEG of the
app's window under `~/.memorizz/desktop/screenshots/<date>/`, and the step's
result ends with the file's path. Connect with `--no-screenshots` to keep only
the text.

To review runs, open `memorizz ui`:

- **Memory Types → Workflows** (`/memory/workflow-memory`) lists every stored
  run with its steps and outcome.
- **Continual Learning** groups repeated runs into trajectory classes. With
  continual learning on, a desktop procedure that keeps succeeding can become
  a [learned skill](../guides/continual-learning.md).
- **Playground** and **Traces** show each tool call of a turn as it ran.

## Use it from Python

Attach the server like any other local MCP server:

```python
import sys

from memorizz import MemAgent

agent = MemAgent(
    model=model,
    memory_provider=provider,
    mcp_servers=[
        {
            "name": "desktop",
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "memorizz", "desktop", "serve"],
            "timeout": 60.0,
            "require_approval": True,
        }
    ],
)

agent.run("Open Calculator and work out 12 × 7.")

for proposal in agent.list_approval_proposals(status="pending"):
    print(proposal["tool_name"], proposal["arguments"])
    agent.approve(proposal["proposal_id"], approver_id="you@example.com")
    print(agent.resume_approval(proposal["proposal_id"]).assistant_response)
```

`memorizz desktop serve` takes `--state PATH` (where the last read is kept) and
`--no-screenshots`. Any MCP host that runs stdio servers can use it too.

## Limits

- **macOS only.** On another platform the tools report that desktop control
  needs macOS.
- **Apps that hide their content.** Some Electron apps expose only an empty
  window frame; Notion's desktop app is one. The agent can still open a Notion
  page there with `open_url`, but it cannot read or click inside it. To read or
  edit Notion pages, use the
  [Notion MCP preset](../guides/mcp-connectivity.md#notion).
- **Web pages.** Chrome shows its toolbar and tabs, but not page content unless
  you turn on `chrome://accessibility`. For websites, use
  [browser control](../browser-control/index.md).
- **One read at a time.** Numbers refer to the latest read of any app; reading
  another app replaces them.

## Troubleshooting

| Message | What to do |
|---|---|
| `Accessibility: not allowed for <app>` | Turn on the app `doctor` names under System Settings → Privacy & Security → Accessibility, then restart it. |
| `Screen Recording: not allowed` | Steps are kept without screenshots. Allow Screen Recording for the same app to keep them. |
| `Desktop control needs pyobjc` | Install the extra: `pip install "memorizz[desktop]"`, or reinstall the uv tool with it. |
| `The window changed since the last read` | Expected after the app changed on its own; the agent reads again and continues. |
| The agent has no `desktop__` tools | Run `/desktop on` in the chat (or restart the chat after `memorizz desktop connect`), and check `memorizz mcp test desktop`. |
