"""Environment-based Notion construction shared by CLI, UI and MCP delivery."""

import os

from .provider import NotionConfig, NotionProvider


def create_notion_provider_from_env(
    *,
    data_source_id=None,
    token=None,
    semantic_backend=None,
    read_only=False,
    embedding_provider=None,
    embedding_config=None,
):
    from ..._env_io import memorizz_home

    data_source_id = data_source_id or os.getenv("MEMORIZZ_NOTION_DATA_SOURCE_ID")
    if not data_source_id:
        raise ValueError(
            "MEMORIZZ_NOTION_DATA_SOURCE_ID is required for the Notion backend"
        )
    # Validate the canonical configuration before opening a vector DB connection.
    config = NotionConfig(
        data_source_id=data_source_id,
        token=token or os.getenv("NOTION_TOKEN"),
        state_path=os.getenv("MEMORIZZ_NOTION_STATE_PATH") or None,
        read_only=read_only,
    )
    backend = (
        str(
            semantic_backend
            or os.getenv("MEMORIZZ_NOTION_SEMANTIC_BACKEND", "filesystem")
        )
        .strip()
        .lower()
    )
    embedding_provider = (
        embedding_provider or os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER") or None
    )
    if isinstance(embedding_provider, str):
        defaults = {}
        model = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL")
        dimensions = os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS")
        if model:
            defaults["model"] = model
        if dimensions:
            defaults["dimensions"] = int(dimensions)
        embedding_config = {**defaults, **(embedding_config or {})}
    semantic = None
    if backend == "filesystem":
        from ..filesystem import FileSystemConfig, FileSystemProvider

        semantic = FileSystemProvider(
            FileSystemConfig(
                root_path=os.getenv("MEMORIZZ_NOTION_VECTOR_PATH")
                or memorizz_home() / "notion-vectors" / config.data_source_id,
                embedding_provider=embedding_provider,
                embedding_config=embedding_config,
                lazy_vector_indexes=True,
            )
        )
    elif backend == "mongodb":
        from ..mongodb import MongoDBConfig, MongoDBProvider

        uri = os.getenv("MEMORIZZ_NOTION_MONGODB_URI") or os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError(
                "MONGODB_URI or MEMORIZZ_NOTION_MONGODB_URI is required for Notion vectors"
            )
        semantic = MongoDBProvider(
            MongoDBConfig(
                uri=uri,
                db_name=os.getenv(
                    "MEMORIZZ_NOTION_VECTOR_DB", "memorizz_notion_vectors"
                ),
                lazy_vector_indexes=True,
                read_only=read_only,
                embedding_provider=embedding_provider,
                embedding_config=embedding_config,
            )
        )
    elif backend == "oracle":
        from ..oracle import OracleProvider

        semantic = OracleProvider.from_env(
            index_policy="lazy",
            in_database_embedding=False,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
        )
    elif backend != "none":
        raise ValueError(
            "Notion semantic backend must be filesystem, mongodb, oracle, or none"
        )
    try:
        provider = NotionProvider(config, semantic_provider=semantic)
        provider._owns_semantic_provider = semantic is not None
        return provider
    except Exception:
        if semantic is not None:
            semantic.close()
        raise
