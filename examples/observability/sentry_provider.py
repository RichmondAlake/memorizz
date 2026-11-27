"""Send a MemAgent's turns to Sentry, and see what is sent.

An agent with one tool answers a question. With ``observability_provider=
"sentry"`` the turn is sent to Sentry as one trace: ``invoke_agent <name>``
with a span per model call, tool call and memory read.

    pip install "memorizz[sentry]"

    # No Sentry account needed: a stand-in for Sentry runs on this machine
    # and the script prints the trace it received.
    python examples/observability/sentry_provider.py

    # No account either, with Sentry's own local server (needs the Sentry
    # CLI): the turn shows in the terminal as it happens.
    sentry local run -- python examples/observability/sentry_provider.py

    # Your Sentry project: the turn appears in Sentry's trace explorer.
    SENTRY_DSN="https://<key>@o<org>.ingest.sentry.io/<project>" \\
    SENTRY_ORG="<your-org-slug>" \\
        python examples/observability/sentry_provider.py

The model is OpenAI's unless you say otherwise:

    PROVIDER=ollama MODEL=qwen2.5:7b python examples/observability/sentry_provider.py

Set ``INCLUDE_CONTENT=1`` to also send the tool's arguments and result.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROVIDER = os.environ.get("PROVIDER", "openai")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
INCLUDE_CONTENT = os.environ.get("INCLUDE_CONTENT", "") not in ("", "0")

RECEIVED: list = []


class _StandInSentry(BaseHTTPRequestHandler):
    """Accepts what the Sentry SDK posts and keeps it, as Sentry's ingest
    endpoint would."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        lines = [line for line in body.split(b"\n") if line.strip()]
        for header, payload in zip(lines[1::2], lines[2::2]):
            RECEIVED.append((json.loads(header).get("type"), json.loads(payload)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def start_stand_in() -> str:
    """Run the stand-in on a free local port and return a DSN that points
    the Sentry SDK at it."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StandInSentry)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://examplekey@127.0.0.1:{server.server_address[1]}/1"


def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


def main() -> None:
    local = os.environ.get("SENTRY_SPOTLIGHT", "").strip()
    has_dsn = bool(
        os.environ.get("MEMORIZZ_SENTRY_DSN") or os.environ.get("SENTRY_DSN")
    )
    stand_in = not has_dsn and not local
    if stand_in:
        os.environ["SENTRY_DSN"] = start_stand_in()
        os.environ.setdefault("NO_PROXY", "127.0.0.1")

    from memorizz.memagent import MemAgent

    agent = MemAgent(
        name="Sentry example",
        instruction="Use the add tool for arithmetic, then answer in one sentence.",
        llm_config={"provider": PROVIDER, "model": MODEL},
        tools=[add],
        memory_provider=False,
        observability=True,
        observability_provider={
            "provider": "sentry",
            "include_content": INCLUDE_CONTENT,
        },
    )

    status = agent.observability_status()
    print(f"Provider: {status['label']}, ready: {status['ready']}")
    if not status["ready"]:
        print("Missing:", ", ".join(status["missing"]))
        return

    answer = agent.run("What is 19 + 23?", trace_tags={"example": "sentry"})
    print("Answer:", answer)

    agent.flush_observability(10)  # wait for the trace to be sent
    root = agent.get_trace_context()["root_trace_id"]
    print("MemoRizz trace:", root)

    if status.get("local"):
        print("Sent to Sentry's local server at", status["local"])
    if has_dsn:
        print("Open it in Sentry:", agent.observability_provider.trace_url(root))
        print("All traces:", agent.observability_url())
    if not stand_in:
        return

    print("\nWhat the stand-in Sentry received:")
    for kind, payload in RECEIVED:
        if kind != "transaction":
            print(f"  {kind}: {payload.get('message')}")
            continue
        trace = payload["contexts"]["trace"]
        print(
            f"  trace {trace['trace_id']}  {payload['transaction']}  [{trace['status']}]"
        )
        usage = {
            key[len("gen_ai.usage.") :]: value
            for key, value in trace["data"].items()
            if key.startswith("gen_ai.usage.")
        }
        print(f"    tokens: {usage}   tags: {payload['tags']}")
        for span in payload["spans"]:
            extra = ""
            if "gen_ai.tool.call.arguments" in span["data"]:
                extra = (
                    f"  args={span['data']['gen_ai.tool.call.arguments']}"
                    f" result={span['data'].get('gen_ai.tool.call.result')}"
                )
            print(f"    {span['op']:<28} {span['description']}{extra}")
    if not RECEIVED:
        print("  nothing: is the sentry extra installed?")
    agent.close()


if __name__ == "__main__":
    main()
