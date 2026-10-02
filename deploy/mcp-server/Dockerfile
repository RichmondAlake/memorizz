# MemoRizz MCP server over Streamable HTTP, for hosting behind HTTPS.
# Build from the repository root:
#   docker build -f deploy/mcp-server/Dockerfile -t memorizz-mcp .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMORIZZ_HOME=/data \
    MEMORIZZ_MEMORY_ROOT=/data/memory \
    MEMORIZZ_MCP_SERVER_TRANSPORT=streamable-http \
    MEMORIZZ_MCP_SERVER_HOST=0.0.0.0 \
    MEMORIZZ_MCP_SERVER_PORT=8766 \
    MEMORIZZ_MCP_SERVER_STATELESS_HTTP=true \
    MEMORIZZ_MCP_SERVER_ALLOW_WRITES=true

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY eval ./eval
COPY plugins ./plugins
RUN pip install ".[mcp]" && useradd --create-home --uid 10001 memorizz \
    && mkdir -p /data && chown memorizz /data

USER memorizz
VOLUME ["/data"]
EXPOSE 8766
# API keys come from MEMORIZZ_MCP_SERVER_API_KEYS ({"alice": "token", ...});
# the public HTTPS origin from MEMORIZZ_MCP_SERVER_PUBLIC_URL.
CMD ["memorizz", "mcp", "serve"]
