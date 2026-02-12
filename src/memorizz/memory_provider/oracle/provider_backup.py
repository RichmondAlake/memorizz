import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import oracledb
except ImportError:
    raise ImportError(
        "oracledb package is required for Oracle provider. "
        "Install it with: pip install oracledb"
    )

from ...enums.memory_type import MemoryType
from ...long_term_memory.semantic.persona.persona import Persona
from ...long_term_memory.semantic.persona.role_type import RoleType
from ...memagent import MemAgentModel
from ..base import MemoryProvider

if TYPE_CHECKING:
    from ...embeddings import get_embedding, get_embedding_dimensions

logger = logging.getLogger(__name__)


@dataclass
class OracleConfig:
    """Configuration for the Oracle provider."""

    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        schema: Optional[str] = None,
        lazy_vector_indexes: bool = False,
        embedding_provider=None,
        embedding_config: Dict[str, Any] = None,
        pool_min: int = 1,
        pool_max: int = 5,
        pool_increment: int = 1,
    ):
        """
        Initialize the Oracle provider with configuration settings.

        Parameters:
        -----------
        user : str
            Oracle database username
        password : str
            Oracle database password
        dsn : str
            Oracle DSN (Data Source Name) or connection string
            Examples:
            - "localhost:1521/FREEPDB1"
            - "myhost.example.com:1521/XEPDB1"
            - Full TNS string
        schema : str, optional
            Schema name for tables (default: None, uses username as schema)
        lazy_vector_indexes : bool
            If True, vector indexes are created only when needed
            If False, vector indexes are created immediately during initialization
            Default: False
        embedding_provider : str or EmbeddingManager, optional
            Embedding provider to use. Can be:
            - EmbeddingManager instance (explicit injection)
            - String provider name ("openai", "ollama", "voyageai")
            - None (uses global embedding configuration)
        embedding_config : Dict[str, Any], optional
            Configuration for the embedding provider
            Example: {"model": "text-embedding-3-small", "dimensions": 512}
        pool_min : int
            Minimum number of connections in the pool (default: 1)
        pool_max : int
            Maximum number of connections in the pool (default: 5)
        pool_increment : int
            Number of connections to add when pool is exhausted (default: 1)
        """
        self.user = user
        self.password = password
        self.dsn = dsn
        self.schema = schema if schema is not None else user
        self.lazy_vector_indexes = lazy_vector_indexes
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config or {}
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.pool_increment = pool_increment


class OracleProvider(MemoryProvider):
    """Oracle Database implementation of the MemoryProvider interface."""

    def __init__(self, config: OracleConfig):
        """
        Initialize the Oracle provider with configuration settings.

        Parameters:
        -----------
        config : OracleConfig
            Configuration object containing connection parameters
        """
        self.config = config

        # Initialize connection pool
        try:
            self.pool = oracledb.create_pool(
                user=config.user,
                password=config.password,
                dsn=config.dsn,
                min=config.pool_min,
                max=config.pool_max,
                increment=config.pool_increment,
            )
            logger.info(f"Oracle connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create Oracle connection pool: {e}")
            raise

        # Track which vector indexes have been created
        self._vector_indexes_created = set()

        # Process embedding provider configuration
        self._embedding_provider = self._setup_embedding_provider(config)

        # Create all memory store tables
        self._create_memory_stores()

        # Create vector indexes immediately only if not using lazy initialization
        if not config.lazy_vector_indexes:
            try:
                self._create_vector_indexes_for_memory_stores()
            except Exception as e:
                logger.warning(
                    f"Failed to create vector indexes during initialization: {e}"
                )
                logger.info("Vector indexes will be created lazily when needed")
                self.config.lazy_vector_indexes = True

    def _setup_embedding_provider(self, config: OracleConfig):
        """Setup the embedding provider based on configuration."""
        if config.embedding_provider is None:
            return None
        elif isinstance(config.embedding_provider, str):
            try:
                from ...embeddings import EmbeddingManager

                provider = EmbeddingManager(
                    config.embedding_provider, config.embedding_config
                )
                logger.info(
                    f"Created embedding provider: {provider.get_provider_info()}"
                )
                return provider
            except Exception as e:
                logger.error(
                    f"Failed to create embedding provider '{config.embedding_provider}': {e}"
                )
                raise
        else:
            return config.embedding_provider

    def _get_embedding_provider(self):
        """Get the embedding provider to use, with fallback logic."""
        if self._embedding_provider is not None:
            return self._embedding_provider
        else:
            from ...embeddings import get_embedding_manager

            return get_embedding_manager()

    def _get_embedding_dimensions_safe(self) -> int:
        """Safely get embedding dimensions with error handling."""
        try:
            if self._embedding_provider is not None:
                return self._embedding_provider.get_dimensions()
            else:
                from ...embeddings import get_embedding_dimensions

                return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Failed to get embedding dimensions: {e}")
            raise RuntimeError(
                "Cannot determine embedding dimensions. Please configure embeddings first using:\n"
                "configure_embeddings('openai', {'model': 'text-embedding-3-small', 'dimensions': 512})\n"
                "Or use lazy_vector_indexes=True to defer vector index creation."
            )

    def _get_connection(self):
        """Get a connection from the pool."""
        return self.pool.acquire()

    def _get_table_name(self, memory_type: MemoryType) -> str:
        """Get the table name for a memory type."""
        return f"{self.config.schema}.{memory_type.value}"

    def _create_memory_stores(self) -> None:
        """Create all memory store tables in Oracle."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get embedding dimensions for vector columns
            try:
                dimensions = self._get_embedding_dimensions_safe()
            except:
                # If dimensions not available yet, use placeholder
                dimensions = 1536  # Common default
                logger.warning(
                    f"Using default dimensions {dimensions} for table creation"
                )

            for memory_type in MemoryType:
                table_name = self._get_table_name(memory_type)

                # Check if table exists
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM user_tables
                    WHERE table_name = UPPER(:1)
                    """,
                    (memory_type.value,),
                )
                exists = cursor.fetchone()[0] > 0

                if not exists:
                    # Create table with appropriate schema
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
                        data CLOB CHECK (data IS JSON),
                        embedding VECTOR({dimensions}, FLOAT32),
                        name VARCHAR2(255),
                        memory_id VARCHAR2(255),
                        agent_id VARCHAR2(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                    try:
                        cursor.execute(create_sql)
                        conn.commit()
                        logger.info(f"Created table: {table_name}")
                    except Exception as e:
                        logger.warning(f"Error creating table {table_name}: {e}")
                        conn.rollback()

            # Create indexes on commonly queried fields
            self._create_standard_indexes(cursor, conn)

    def _create_standard_indexes(self, cursor, conn):
        """Create standard B-tree indexes on commonly queried fields."""
        index_definitions = [
            ("name", ["name"]),
            ("memory_id", ["memory_id"]),
            ("agent_id", ["agent_id"]),
            ("created_at", ["created_at"]),
        ]

        for memory_type in MemoryType:
            table_name = self._get_table_name(memory_type)

            for idx_name, columns in index_definitions:
                full_idx_name = f"idx_{memory_type.value}_{idx_name}"
                try:
                    cursor.execute(
                        f"""
                        CREATE INDEX {full_idx_name}
                        ON {table_name} ({', '.join(columns)})
                        """
                    )
                    conn.commit()
                except Exception as e:
                    # Index might already exist
                    conn.rollback()

    def _create_vector_indexes_for_memory_stores(self) -> None:
        """Create vector indexes for all memory stores."""
        for memory_type in MemoryType:
            try:
                self._ensure_vector_index(memory_type)
            except Exception as e:
                logger.warning(
                    f"Failed to create vector index for {memory_type.value}: {e}"
                )

    def _ensure_vector_index(self, memory_type: MemoryType):
        """Ensure vector index exists for a memory type."""
        index_key = f"{memory_type.value}_vector_index"

        if index_key in self._vector_indexes_created:
            return

        table_name = self._get_table_name(memory_type)
        index_name = f"idx_{memory_type.value}_vec"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if index exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM user_indexes
                WHERE index_name = UPPER(:1)
                """,
                (index_name,),
            )
            exists = cursor.fetchone()[0] > 0

            if not exists:
                try:
                    # Create vector index
                    # Oracle 23ai+ supports HNSW and IVF indexes
                    cursor.execute(
                        f"""
                        CREATE VECTOR INDEX {index_name}
                        ON {table_name} (embedding)
                        ORGANIZATION INMEMORY NEIGHBOR GRAPH
                        DISTANCE COSINE
                        WITH TARGET ACCURACY 95
                        """
                    )
                    conn.commit()
                    logger.info(f"Created vector index: {index_name}")
                    self._vector_indexes_created.add(index_key)
                except Exception as e:
                    logger.warning(f"Failed to create vector index {index_name}: {e}")
                    conn.rollback()
            else:
                self._vector_indexes_created.add(index_key)

    def _doc_to_dict(
        self, doc_data: str, include_embedding: bool = False
    ) -> Dict[str, Any]:
        """Convert JSON document to dictionary."""
        if not doc_data:
            return {}
        result = json.loads(doc_data)
        if not include_embedding and "embedding" in result:
            del result["embedding"]
        return result

    def _dict_to_doc(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to JSON document."""
        return json.dumps(data)

    def store(
        self,
        data: Dict[str, Any] = None,
        memory_store_type: MemoryType = None,
        memory_id: str = None,
        memory_unit: Any = None,
    ) -> str:
        """
        Store data in Oracle.

        Parameters:
        -----------
        data : Dict[str, Any], optional
            The document to be stored (legacy parameter)
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_id : str, optional
            The memory ID to associate with (new parameter)
        memory_unit : MemoryUnit, optional
            The memory unit object to store (new parameter)

        Returns:
        --------
        str
            The ID of the inserted/updated document
        """
        # Handle new calling style (memory_unit + memory_id)
        if memory_unit is not None:
            # Convert memory_unit to dict
            if hasattr(memory_unit, "model_dump"):
                data = memory_unit.model_dump()
            elif hasattr(memory_unit, "dict"):
                data = memory_unit.dict()
            else:
                data = memory_unit.__dict__

            # Add memory_id if provided
            if memory_id:
                data["memory_id"] = memory_id

            # Determine memory_store_type from memory_unit
            if hasattr(memory_unit, "memory_type"):
                memory_store_type = memory_unit.memory_type
            elif "memory_type" in data:
                memory_store_type = data["memory_type"]
            else:
                memory_store_type = MemoryType.CONVERSATION_MEMORY

        # Validate we have required parameters
        if data is None or memory_store_type is None:
            raise ValueError(
                "Either (data, memory_store_type) or (memory_unit) must be provided"
            )

        # Ensure memory_store_type is MemoryType enum
        if isinstance(memory_store_type, str):
            memory_store_type = MemoryType(memory_store_type)

        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Extract key fields
            doc_id = data.get("_id")
            name = data.get("name")
            memory_id = data.get("memory_id")
            agent_id = data.get("agent_id")
            embedding = data.get("embedding")

            # Store full data as JSON
            data_copy = data.copy()
            doc_json = self._dict_to_doc(data_copy)

            if doc_id:
                # Update existing document
                # Convert string UUID to RAW
                if isinstance(doc_id, str):
                    doc_id = uuid.UUID(doc_id).bytes

                update_sql = f"""
                UPDATE {table_name}
                SET data = :data,
                    name = :name,
                    memory_id = :memory_id,
                    agent_id = :agent_id,
                    embedding = :embedding,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                """
                cursor.execute(
                    update_sql,
                    {
                        "data": doc_json,
                        "name": name,
                        "memory_id": memory_id,
                        "agent_id": agent_id,
                        "embedding": embedding,
                        "id": doc_id,
                    },
                )

                if cursor.rowcount == 0:
                    # Document doesn't exist, insert it
                    insert_sql = f"""
                    INSERT INTO {table_name} (id, data, name, memory_id, agent_id, embedding)
                    VALUES (:id, :data, :name, :memory_id, :agent_id, :embedding)
                    """
                    cursor.execute(
                        insert_sql,
                        {
                            "id": doc_id,
                            "data": doc_json,
                            "name": name,
                            "memory_id": memory_id,
                            "agent_id": agent_id,
                            "embedding": embedding,
                        },
                    )

                conn.commit()
                return str(uuid.UUID(bytes=doc_id))
            else:
                # Insert new document
                new_id = uuid.uuid4().bytes
                insert_sql = f"""
                INSERT INTO {table_name} (id, data, name, memory_id, agent_id, embedding)
                VALUES (:id, :data, :name, :memory_id, :agent_id, :embedding)
                """
                cursor.execute(
                    insert_sql,
                    {
                        "id": new_id,
                        "data": doc_json,
                        "name": name,
                        "memory_id": memory_id,
                        "agent_id": agent_id,
                        "embedding": embedding,
                    },
                )
                conn.commit()
                return str(uuid.UUID(bytes=new_id))

    def retrieve_by_query(
        self,
        query: Union[Dict[str, Any], str],
        memory_store_type: MemoryType = None,
        limit: int = 1,
        include_embedding: bool = False,
        memory_id: str = None,
        memory_type: Union[str, MemoryType] = None,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve documents from Oracle by query.

        Parameters:
        -----------
        query : Union[Dict[str, Any], str]
            The query to use for retrieval
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_type : Union[str, MemoryType], optional
            The type of memory store (new parameter, takes precedence)
        memory_id : str, optional
            Filter results to specific memory_id
        limit : int
            The maximum number of documents to return
        include_embedding : bool
            Whether to include the embedding field in the results

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved documents, or None if not found
        """
        # Handle new calling style: memory_type takes precedence over memory_store_type
        if memory_type is not None:
            if isinstance(memory_type, str):
                memory_store_type = MemoryType(memory_type)
            else:
                memory_store_type = memory_type

        if memory_store_type is None:
            raise ValueError("Either memory_store_type or memory_type must be provided")

        # If memory_id filter is provided, add it to the query
        if memory_id is not None:
            if isinstance(query, dict):
                query = {**query, "memory_id": memory_id}
            else:
                # For string queries (semantic search), store memory_id for filtering
                kwargs["memory_id"] = memory_id

        # Handle special cases with vector search
        if memory_store_type == MemoryType.PERSONAS:
            return self.retrieve_persona_by_query(query, limit=limit)
        elif memory_store_type == MemoryType.TOOLBOX:
            return self.retrieve_toolbox_item(query, limit)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.retrieve_workflow_by_query(query, limit)
        elif memory_store_type == MemoryType.SUMMARIES:
            return self.retrieve_summaries_by_query(query, limit)
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            if isinstance(query, dict):
                # Filter query for loading existing cache entries
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            else:
                # String query for semantic similarity search
                return self.find_similar_cache_entries(query, limit=limit, **kwargs)
        else:
            # Standard query
            if isinstance(query, dict):
                return self._retrieve_by_filter(
                    query, memory_store_type, limit, include_embedding
                )
            else:
                return []

    def _retrieve_by_filter(
        self,
        query: Dict[str, Any],
        memory_store_type: MemoryType,
        limit: int,
        include_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents by filter criteria."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build WHERE clause from query
            where_clauses = []
            params = {}

            for key, value in query.items():
                if key != "_id":
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

            sql = f"""
            SELECT id, data, embedding
            FROM {table_name}
            WHERE {where_sql}
            FETCH FIRST :limit ROWS ONLY
            """

            params["limit"] = limit
            cursor.execute(sql, params)

            results = []
            for row in cursor:
                doc = self._doc_to_dict(row[1], include_embedding)
                doc["_id"] = str(uuid.UUID(bytes=row[0]))
                if include_embedding and row[2] is not None:
                    doc["embedding"] = list(row[2])
                results.append(doc)

            return results if results else None

    def retrieve_by_id(
        self, id: str, memory_store_type: MemoryType
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from Oracle by ID.

        Parameters:
        -----------
        id : str
            The document ID
        memory_store_type : MemoryType
            The type of memory store

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found
        """
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                doc_id = uuid.UUID(id).bytes
            except:
                return None

            cursor.execute(
                f"""
                SELECT id, data
                FROM {table_name}
                WHERE id = :id
                """,
                {"id": doc_id},
            )

            row = cursor.fetchone()
            if row:
                doc = self._doc_to_dict(row[1])
                doc["_id"] = str(uuid.UUID(bytes=row[0]))
                return doc

            return None

    def retrieve_by_name(
        self, name: str, memory_store_type: MemoryType, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document by name."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT id, data, embedding
                FROM {table_name}
                WHERE name = :name
                FETCH FIRST 1 ROWS ONLY
                """,
                {"name": name},
            )

            row = cursor.fetchone()
            if row:
                doc = self._doc_to_dict(row[1], include_embedding)
                doc["_id"] = str(uuid.UUID(bytes=row[0]))
                if include_embedding and row[2] is not None:
                    doc["embedding"] = list(row[2])
                return doc

            return None

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """Delete a document by ID."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                doc_id = uuid.UUID(id).bytes
            except:
                return False

            cursor.execute(f"DELETE FROM {table_name} WHERE id = :id", {"id": doc_id})
            conn.commit()

            return cursor.rowcount > 0

    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """Delete a document by name."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"DELETE FROM {table_name} WHERE name = :name", {"name": name}
            )
            conn.commit()

            return cursor.rowcount > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """Delete all documents within a memory store type."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()

            return cursor.rowcount > 0

    def list_all(
        self, memory_store_type: MemoryType, include_embedding: bool = False
    ) -> List[Dict[str, Any]]:
        """List all documents within a memory store type."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT id, data, embedding
                FROM {table_name}
                """
            )

            results = []
            for row in cursor:
                doc = self._doc_to_dict(row[1], include_embedding)
                doc["_id"] = str(uuid.UUID(bytes=row[0]))
                if include_embedding and row[2] is not None:
                    doc["embedding"] = list(row[2])
                results.append(doc)

            return results

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        include_embedding: bool = False,
        memory_type: Union[str, MemoryType] = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The memory ID to retrieve history for
        include_embedding : bool
            Whether to include embeddings
        memory_type : Union[str, MemoryType], optional
            Type of memory (defaults to CONVERSATION_MEMORY)
        limit : int, optional
            Maximum number of entries to return
        """
        # Default to CONVERSATION_MEMORY if not specified
        if memory_type is None:
            memory_type = MemoryType.CONVERSATION_MEMORY
        elif isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        table_name = self._get_table_name(memory_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build SQL with optional limit
            sql = f"""
                SELECT id, data, embedding
                FROM {table_name}
                WHERE memory_id = :memory_id
                ORDER BY JSON_VALUE(data, '$.timestamp')
            """

            if limit is not None:
                sql += f" FETCH FIRST :limit ROWS ONLY"
                cursor.execute(sql, {"memory_id": memory_id, "limit": limit})
            else:
                cursor.execute(sql, {"memory_id": memory_id})

            results = []
            for row in cursor:
                doc = self._doc_to_dict(row[1], include_embedding)
                doc["_id"] = str(uuid.UUID(bytes=row[0]))
                if include_embedding and row[2] is not None:
                    doc["embedding"] = list(row[2])
                results.append(doc)

            return results

    def update_by_id(
        self, id: str, data: Dict[str, Any], memory_store_type: MemoryType
    ) -> bool:
        """Update a document by ID."""
        table_name = self._get_table_name(memory_store_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                doc_id = uuid.UUID(id).bytes
            except:
                return False

            # Get existing document
            cursor.execute(
                f"SELECT data FROM {table_name} WHERE id = :id", {"id": doc_id}
            )
            row = cursor.fetchone()

            if not row:
                return False

            # Merge with existing data
            existing_doc = self._doc_to_dict(row[0])
            existing_doc.update(data)

            # Extract fields
            name = existing_doc.get("name")
            memory_id = existing_doc.get("memory_id")
            agent_id = existing_doc.get("agent_id")
            embedding = existing_doc.get("embedding")

            doc_json = self._dict_to_doc(existing_doc)

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET data = :data,
                    name = :name,
                    memory_id = :memory_id,
                    agent_id = :agent_id,
                    embedding = :embedding,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
                """,
                {
                    "data": doc_json,
                    "name": name,
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "embedding": embedding,
                    "id": doc_id,
                },
            )
            conn.commit()

            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the connection pool."""
        if hasattr(self, "pool"):
            self.pool.close()
            logger.info("Oracle connection pool closed")

    # ===== VECTOR SEARCH METHODS =====

    def retrieve_persona_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve personas using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(MemoryType.PERSONAS, embedding, limit=limit)

    def retrieve_toolbox_item(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve toolbox items using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(MemoryType.TOOLBOX, embedding, limit=limit)

    def retrieve_workflow_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve workflows using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(MemoryType.WORKFLOW_MEMORY, embedding, limit=limit)

    def retrieve_summaries_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve summaries using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        return self._vector_search(MemoryType.SUMMARIES, embedding, limit=limit)

    def find_similar_cache_entries(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Find semantically similar cache entries using vector search."""
        from ...embeddings import get_embedding

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for semantic cache query: {e}")
            return []

        # Extract filters
        filters = {}
        if "agent_id" in kwargs and kwargs["agent_id"] is not None:
            filters["agent_id"] = kwargs["agent_id"]
        if "memory_id" in kwargs and kwargs["memory_id"] is not None:
            filters["memory_id"] = kwargs["memory_id"]
        if "session_id" in kwargs and kwargs["session_id"] is not None:
            filters["session_id"] = kwargs["session_id"]

        return self._vector_search(
            MemoryType.SEMANTIC_CACHE, embedding, limit=limit, filters=filters
        )

    def _vector_search(
        self,
        memory_type: MemoryType,
        query_embedding: List[float],
        limit: int = 5,
        filters: Dict[str, Any] = None,
        memory_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using Oracle's VECTOR_DISTANCE function.

        Parameters:
        -----------
        memory_type : MemoryType
            The type of memory store to search
        query_embedding : List[float]
            The query embedding vector
        limit : int
            Maximum number of results to return
        filters : Dict[str, Any]
            Additional filter criteria
        memory_id : str
            Memory ID to filter by

        Returns:
        --------
        List[Dict[str, Any]]
            List of similar documents with scores
        """
        # Ensure vector index exists (lazy creation)
        if self.config.lazy_vector_indexes:
            self._ensure_vector_index(memory_type)

        table_name = self._get_table_name(memory_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build filter clauses
            filter_clauses = []
            params = {"limit": limit}

            if memory_id:
                filter_clauses.append("memory_id = :memory_id")
                params["memory_id"] = memory_id

            if filters:
                for key, value in filters.items():
                    filter_clauses.append(f"{key} = :{key}")
                    params[key] = value

            where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"

            # Oracle vector search using VECTOR_DISTANCE
            # Note: Oracle returns distance (lower is better), we convert to score
            sql = f"""
            SELECT
                id,
                data,
                (1 - VECTOR_DISTANCE(embedding, :query_vec, COSINE)) as score
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY VECTOR_DISTANCE(embedding, :query_vec, COSINE)
            FETCH FIRST :limit ROWS ONLY
            """

            params["query_vec"] = query_embedding

            try:
                cursor.execute(sql, params)

                results = []
                for row in cursor:
                    doc = self._doc_to_dict(row[1])
                    doc["_id"] = str(uuid.UUID(bytes=row[0]))
                    doc["score"] = float(row[2])
                    results.append(doc)

                return results
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    # ===== MEMAGENT METHODS =====

    def store_memagent(self, memagent: "MemAgentModel") -> str:
        """Store a memagent in the Oracle database."""
        memagent_dict = memagent.model_dump()

        # Remove agent_id field since we only want to use _id
        memagent_dict.pop("agent_id", None)

        # Convert persona to serializable format
        if memagent.persona:
            memagent_dict["persona"] = memagent.persona.to_dict()

        # Remove function objects from tools
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]

        # Store using the standard store method
        doc_id = self.store(memagent_dict, MemoryType.MEMAGENT)
        return doc_id

    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """Delete a memagent from the memory provider."""
        if cascade:
            memagent = self.retrieve_memagent(agent_id)
            if memagent is None:
                raise ValueError(f"MemAgent with id {agent_id} not found")

            # Delete all memory units associated with the memagent
            for memory_id in memagent.memory_ids:
                for memory_type in MemoryType:
                    self._delete_memory_units_by_memory_id(memory_id, memory_type)

        return self.delete_by_id(agent_id, MemoryType.MEMAGENT)

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Update the memory_ids of a memagent."""
        return self.update_by_id(
            agent_id, {"memory_ids": memory_ids}, MemoryType.MEMAGENT
        )

    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """Delete the memory_ids of a memagent."""
        return self.update_by_id(agent_id, {"memory_ids": []}, MemoryType.MEMAGENT)

    def list_memagents(self) -> List["MemAgentModel"]:
        """List all memagents in the Oracle database."""
        documents = self.list_all(MemoryType.MEMAGENT)
        agents = []

        for doc in documents:
            agent = MemAgentModel(
                name=doc.get("name"),
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=str(doc.get("_id")),
                tools=doc.get("tools"),
                long_term_memory_ids=doc.get("long_term_memory_ids"),
                memory_provider=self,
            )

            # Construct persona if present
            if doc.get("persona"):
                persona_data = doc.get("persona")
                role_str = persona_data.get("role")
                role = None

                for role_type in RoleType:
                    if role_type.value == role_str:
                        role = role_type
                        break

                if role is None:
                    role = RoleType.GENERAL

                agent.persona = Persona(
                    name=persona_data.get("name"),
                    role=role,
                    goals=persona_data.get("goals"),
                    background=persona_data.get("background"),
                    persona_id=persona_data.get("persona_id"),
                )

            agents.append(agent)

        return agents

    def retrieve_memagent(self, agent_id: str) -> "MemAgentModel":
        """Retrieve a memagent from the Oracle database."""
        document = self.retrieve_by_id(agent_id, MemoryType.MEMAGENT)

        if not document:
            return None

        memagent = MemAgentModel(
            name=document.get("name"),
            instruction=document.get("instruction"),
            application_mode=document.get("application_mode", "assistant"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=str(document.get("_id")),
            tools=document.get("tools"),
            long_term_memory_ids=document.get("long_term_memory_ids"),
            memory_provider=self,
        )

        # Construct persona if present
        if document.get("persona"):
            persona_data = document.get("persona")
            role_str = persona_data.get("role")
            role = None

            for role_type in RoleType:
                if role_type.value == role_str:
                    role = role_type
                    break

            if role is None:
                role = RoleType.GENERAL

            memagent.persona = Persona(
                name=persona_data.get("name"),
                role=role,
                goals=persona_data.get("goals"),
                background=persona_data.get("background"),
                persona_id=persona_data.get("persona_id"),
            )

        return memagent

    def _delete_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ):
        """Delete all memory units associated with a memory_id."""
        table_name = self._get_table_name(memory_type)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {table_name} WHERE memory_id = :memory_id",
                {"memory_id": memory_id},
            )
            conn.commit()
