import time
import logging
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.encryption import ClientEncryption, Algorithm
from bson.binary import Binary, STANDARD
from bson.codec_options import CodecOptions

from ..base import MemoryProvider
from dataclasses import dataclass
from ...enums.memory_type import MemoryType
from ...memagent import MemAgentModel
from ...long_term_memory.semantic.persona.persona import Persona
from ...long_term_memory.semantic.persona.role_type import RoleType
from typing import Dict, Any, Optional, List
from pymongo.operations import SearchIndexModel
from ...embeddings import get_embedding, get_embedding_dimensions

import uuid

logger = logging.getLogger(__name__)

@dataclass
class MongoDBConfig():
    """Configuration for the MongoDB provider."""

    def __init__(self, uri: str, db_name: str = "memorizz", lazy_vector_indexes: bool = False, embedding_provider = None, embedding_config: Dict[str, Any] = None, encryption_config: Dict[str, Any] = None):
        """
        Initialize the MongoDB provider with configuration settings.
        
        Parameters:
        -----------
        uri : str
            The MongoDB URI.
        db_name : str
            The database name.
        lazy_vector_indexes : bool
            If True, vector indexes are created only when needed (when vector operations are performed).
            If False, vector indexes are created immediately during initialization (requires embedding configuration).
            Default: False (maintains backward compatibility)
        embedding_provider : str or EmbeddingManager, optional
            Embedding provider to use. Can be:
            - EmbeddingManager instance (explicit injection)
            - String provider name ("openai", "ollama", "voyageai") 
            - None (uses global embedding configuration)
        embedding_config : Dict[str, Any], optional
            Configuration for the embedding provider. Only used when embedding_provider is a string.
            Example: {"model": "text-embedding-3-small", "dimensions": 512}
        encryption_config : Dict[str, Any], optional
            Configuration for Client-Side Field Level Encryption (CSFLE) settings.
        """
        self.uri = uri
        self.db_name = db_name
        self.lazy_vector_indexes = lazy_vector_indexes
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config or {}
        # The encryption configuration is stored, but no actions are performed.
        self.encryption_config = encryption_config or {}


class MongoDBProvider(MemoryProvider):
    """MongoDB implementation of the MemoryProvider interface."""
    
    def __init__(self, config: MongoDBConfig):
        """
        Initialize the MongoDB provider with configuration settings.

        Parameters:
        -----------
        config : MongoDBConfig
            Configuration object containing database, embedding, and encryption settings.
        """
        self.config = config
        self.client = MongoClient(config.uri)
        self.db = self.client[config.db_name]

        # --- Initialize CSFLE with smart checks and dynamic keymap creation ---
        self._client_encryption = None
        self.keymap = {}  # Initialize the keymap as an empty dictionary

        if self.config.encryption_config:
            try:
                # Extract key vault details from the config
                kms_providers = self.config.encryption_config.get("kms_providers")
                key_vault_namespace = self.config.encryption_config.get("key_vault_namespace")
                key_vault_client = self.config.encryption_config.get("key_vault_client")

                """
                A separate keyvault client is a good idea for encryption because it enforces
                a crucial security practice: the separation of data and keys.
                """

                if not all([kms_providers, key_vault_namespace, key_vault_client]):
                    raise ValueError("Incomplete encryption_config provided.")

                # Check if key vault collection exists before creating it
                key_vault_db, key_vault_coll = key_vault_namespace.split(".", 1)
                if key_vault_coll not in key_vault_client[key_vault_db].list_collection_names():
                    print(f"Key vault collection '{key_vault_coll}' not found. Creating it now.")
                    key_vault_client[key_vault_db].create_collection(key_vault_coll)
                else:
                    print(f"Key vault collection '{key_vault_coll}' already exists.")

                # Initialize the ClientEncryption object
                self._client_encryption = ClientEncryption(
                    kms_providers=kms_providers,
                    key_vault_namespace=key_vault_namespace,
                    key_vault_client=key_vault_client,
                    codec_options=CodecOptions(uuid_representation=STANDARD),
                )

                # Loop through field mappings to get or create data keys
                field_mappings = self.config.encryption_config.get("field_mappings", {})
                for collection_name, mapping in field_mappings.items():
                    # Check if a data key with this alt name already exists
                    data_key = self._client_encryption.get_key_by_alt_name(collection_name)
                    if data_key is None:
                        # If not, create a new one using a UUID as the alt name for uniqueness
                        alt_name = str(uuid.uuid4())
                        data_key_id = self._client_encryption.create_data_key("local", key_alt_names=[alt_name])
                        print(f"Generated new data key with ID: {data_key_id} for collection '{collection_name}'")
                        self.keymap[collection_name] = alt_name
                    else:
                        print(f"Using existing data key for collection '{collection_name}'")
                        # You would need to retrieve the alt name from the data_key document
                        # as it may not match the collection_name. For simplicity, we assume
                        # it does or use the first one available.
                        self.keymap[collection_name] = data_key['keyAltNames'][0]
                        
            except Exception as e:
                logger.error(f"Error initializing CSFLE: {e}")
                self._client_encryption = None
                self.keymap = {}

        # --- EXISTING CODE ---
        self.persona_collection = self.db[MemoryType.PERSONAS.value]
        self.toolbox_collection = self.db[MemoryType.TOOLBOX.value]
        self.short_term_memory_collection = self.db[MemoryType.SHORT_TERM_MEMORY.value]
        self.long_term_memory_collection = self.db[MemoryType.LONG_TERM_MEMORY.value]
        self.conversation_memory_collection = self.db[MemoryType.CONVERSATION_MEMORY.value]
        self.workflow_memory_collection = self.db[MemoryType.WORKFLOW_MEMORY.value]
        self.memagent_collection = self.db[MemoryType.MEMAGENT.value]
        self.shared_memory_collection = self.db[MemoryType.SHARED_MEMORY.value]
        self.summaries_collection = self.db[MemoryType.SUMMARIES.value]

        # Track which vector indexes have been created
        self._vector_indexes_created = set()

        # Process embedding provider configuration
        self._embedding_provider = self._setup_embedding_provider(config)

        # Create all memory stores in MongoDB.
        self._create_memory_stores()

        # Create vector indexes immediately only if not using lazy initialization
        if not config.lazy_vector_indexes:
            try:
                self._create_vector_indexes_for_memory_stores()
            except Exception as e:
                logger.warning(f"Failed to create vector indexes during initialization: {e}")
                logger.info("Vector indexes will be created lazily when needed")
                # Set lazy mode if immediate creation fails
                self.config.lazy_vector_indexes = True


    def _setup_embedding_provider(self, config: MongoDBConfig):
        """
        Setup the embedding provider based on configuration.
        
        Parameters:
        -----------
        config : MongoDBConfig
            The MongoDB configuration
            
        Returns:
        --------
        EmbeddingManager or None
            The configured embedding provider, or None to use global configuration
        """
        if config.embedding_provider is None:
            # No explicit provider - will use global configuration
            return None
        elif isinstance(config.embedding_provider, str):
            # String provider name - create EmbeddingManager
            try:
                from ...embeddings import EmbeddingManager
                provider = EmbeddingManager(config.embedding_provider, config.embedding_config)
                logger.info(f"Created embedding provider: {provider.get_provider_info()}")
                return provider
            except Exception as e:
                logger.error(f"Failed to create embedding provider '{config.embedding_provider}': {e}")
                raise
        else:
            # Assume it's already an EmbeddingManager instance
            return config.embedding_provider

    def _get_embedding_provider(self):
        """
        Get the embedding provider to use, with fallback logic.
        
        Returns:
        --------
        EmbeddingManager or function
            The embedding provider to use
        """
        if self._embedding_provider is not None:
            # Use explicitly provided embedding provider
            return self._embedding_provider
        else:
            # Fall back to global embedding configuration
            from ...embeddings import get_embedding_manager
            return get_embedding_manager()
    
    def _get_embedding_dimensions_safe(self) -> int:
        """
        Safely get embedding dimensions with error handling.
        
        Returns:
        --------
        int
            The embedding dimensions, or None if not available
        """
        try:
            if self._embedding_provider is not None:
                # Use explicit provider
                return self._embedding_provider.get_dimensions()
            else:
                # Use global configuration
                from ...embeddings import get_embedding_dimensions
                return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Failed to get embedding dimensions: {e}")
            raise RuntimeError(
                "Cannot determine embedding dimensions. Please configure embeddings first using:\n"
                "configure_embeddings('openai', {'model': 'text-embedding-3-small', 'dimensions': 512})\n"
                "Or use lazy_vector_indexes=True to defer vector index creation."
            )
    
    def _ensure_vector_index_for_collection(self, collection, collection_name: str, memory_store: bool = False):
        """
        Ensure vector index exists for a collection, creating it lazily if needed.
        
        Parameters:
        -----------
        collection : pymongo.Collection
            The MongoDB collection
        collection_name : str  
            Name of the collection (for tracking)
        memory_store : bool
            Whether this is a memory store collection
        """
        index_key = f"{collection_name}_vector_index"
        
        if index_key not in self._vector_indexes_created:
            try:
                self._setup_vector_search_index(collection, "vector_index", memory_store)
                self._vector_indexes_created.add(index_key)
                logger.info(f"Created vector index for collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to create vector index for {collection_name}: {e}")
                raise

    def _create_memory_stores(self) -> None:
        """
        Create all memory stores in MongoDB.
        """
        self._create_memory_store(MemoryType.MEMAGENT)
        self._create_memory_store(MemoryType.PERSONAS)
        self._create_memory_store(MemoryType.TOOLBOX)
        self._create_memory_store(MemoryType.SHORT_TERM_MEMORY)
        self._create_memory_store(MemoryType.LONG_TERM_MEMORY)
        self._create_memory_store(MemoryType.CONVERSATION_MEMORY)
        self._create_memory_store(MemoryType.WORKFLOW_MEMORY)
        self._create_memory_store(MemoryType.SHARED_MEMORY)
        self._create_memory_store(MemoryType.SUMMARIES)
    
    def _create_memory_store(self, memory_store_type: MemoryType) -> None:
        """
        Create a new memory store in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store to create.

        Returns:
        --------
        None
        """

        # Create collection if it doesn't exist within the database/memory provider
        # Check if the collection exists within the database and if it doesn't, create an empty collection
        for memory_store_type in MemoryType:
            if memory_store_type.value not in self.db.list_collection_names():
                self.db.create_collection(memory_store_type.value)
                print(f"Created collection: {memory_store_type.value} successfully.")
        

    def _create_vector_indexes_for_memory_stores(self) -> None:
        """
        Create a vector index for each memory store in MongoDB.

        Returns:
        --------
        None
        """
        # Create vector indexes for all memory store types
        for memory_store_type in MemoryType:
            # PERSONAS collection doesn't need memory_id filter since it's not memory-scoped
            memory_store_present = memory_store_type != MemoryType.PERSONAS

            self._ensure_vector_index(
                collection=self.db[memory_store_type.value],
                index_name="vector_index",
                memory_store=memory_store_present,
            )
            
    def _is_encrypted_field(self, collection_name: str, field_name: str) -> Optional[str]:
        """
        Checks if a field should be encrypted based on the encryption config.

        Parameters:
        -----------
        collection_name : str
            The name of the collection being checked.
        field_name : str
            The name of the field to check for encryption.

        Returns:
        --------
        Optional[str]
            The encryption algorithm as a string if the field should be encrypted, otherwise None.
        """
        if not self.config.encryption_config:
            return None
        
        field_mappings = self.config.encryption_config.get("field_mappings", {})
        collection_config = field_mappings.get(collection_name)
        
        if not collection_config:
            return None
            
        encrypted_fields = collection_config.get("encrypted_fields", {})
        
        # This now returns the algorithm type from the config
        return encrypted_fields.get(field_name)

    def store(self, data: Dict[str, Any], memory_store_type: MemoryType) -> str:
        """
        Store data in MongoDB, encrypting sensitive fields if CSFLE is configured.

        Parameters:
        -----------
        data : Dict[str, Any]
            The document to be stored.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        str
            The ID of the inserted/updated document (MongoDB _id).
        """
        collection = None
        collection_name = memory_store_type.value

        if memory_store_type == MemoryType.PERSONAS:
            collection = self.persona_collection
        elif memory_store_type == MemoryType.TOOLBOX:
            collection = self.toolbox_collection
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            collection = self.workflow_memory_collection
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            collection = self.short_term_memory_collection
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            collection = self.long_term_memory_collection
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            collection = self.conversation_memory_collection
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            collection = self.shared_memory_collection
        elif memory_store_type == MemoryType.SUMMARIES:
            collection = self.summaries_collection

        if collection is None:
            raise ValueError(f"Invalid memory store type: {memory_store_type}")

        data_copy = data.copy()
        
        # Clean custom ID fields
        custom_id_fields = [
            "persona_id", "tool_id", "workflow_id", "short_term_memory_id",
            "agent_id"
        ]
        if memory_store_type != MemoryType.CONVERSATION_MEMORY:
            custom_id_fields.append("conversation_id")
        if memory_store_type != MemoryType.LONG_TERM_MEMORY:
            custom_id_fields.append("long_term_memory_id")
            
        for field in custom_id_fields:
            data_copy.pop(field, None)

        # Encryption Logic: check if CSFLE is configured and a keymap exists
        if self._client_encryption and self.keymap:
            for field_name, value in list(data_copy.items()):
                algorithm = self._is_encrypted_field(collection_name, field_name)
                
                # If encryption is required for this field...
                if algorithm:
                    key_alt_name = self.keymap.get(collection_name)
                    if not key_alt_name:
                        raise ValueError(f"Encryption key not found for collection: {collection_name}")
                    
                    # Explicitly encrypt the field's value
                    encrypted_value = self._client_encryption.encrypt(
                        value,
                        Algorithm(algorithm),
                        key_alt_name=key_alt_name
                    )
                    data_copy[field_name] = encrypted_value

        # If document has MongoDB _id, update it (upsert)
        if "_id" in data_copy:
            result = collection.update_one(
                {"_id": data_copy["_id"]},
                {"$set": data_copy},
                upsert=True
            )
            return str(data_copy["_id"])
        else:
            # For new documents, let MongoDB generate _id automatically
            result = collection.insert_one(data_copy)
            return str(result.inserted_id)


    def retrieve_by_query(self, query: Dict[str, Any], memory_store_type: MemoryType, limit: int = 1, include_embedding: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve documents from MongoDB, decrypting fields if CSFLE is configured.
        
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of documents to return.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.
        
        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            A list of retrieved and decrypted documents, or None if not found.
        """
        
        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}
        
        documents = []

        # Dispatch logic to call the correct specialized retrieval method
        if memory_store_type == MemoryType.PERSONAS:
            documents = self.retrieve_persona_by_query(query, limit=limit)
        elif memory_store_type == MemoryType.TOOLBOX:
            documents = self.retrieve_toolbox_item(query, limit=limit)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            documents = self.retrieve_workflow_by_query(query, limit=limit)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            documents = list(self.short_term_memory_collection.find(query, projection).limit(limit))
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            documents = list(self.long_term_memory_collection.find(query, projection).limit(limit))
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            documents = list(self.conversation_memory_collection.find(query, projection).limit(limit))
        elif memory_store_type == MemoryType.SUMMARIES:
            documents = self.retrieve_summaries_by_query(query, limit=limit)
        
        # Return early if no documents are found by the dispatcher
        if not documents:
            return None
        
        # --- New Decryption Logic ---
        # Only attempt decryption if CSFLE is enabled
        if self._client_encryption:
            decrypted_documents = []
            for doc in documents:
                decrypted_doc = doc.copy()
                for field_name, value in doc.items():
                    # Encrypted fields are stored as BSON Binary subtype 6
                    if isinstance(value, Binary):
                        try:
                            # Attempt to decrypt the value
                            decrypted_value = self._client_encryption.decrypt(value)
                            decrypted_doc[field_name] = decrypted_value
                        except PyMongoError as e:
                            logger.error(f"Failed to decrypt field '{field_name}' in document {doc.get('_id', '')}: {e}")
                            # Keep the encrypted value if decryption fails to avoid crashing
                            decrypted_doc[field_name] = value  
                decrypted_documents.append(decrypted_doc)
            return decrypted_documents

        # If CSFLE is not configured, just return the raw documents
        return documents
        
    def retrieve_by_id(self, id: str, memory_store_type: MemoryType) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by _id.

        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        # Get the appropriate collection
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.LONG_TERM_MEMORY: self.long_term_memory_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection
        }
        
        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            return None
            
        # Set projection to exclude embedding for performance
        projection = {"embedding": 0} if memory_store_type in [
            MemoryType.PERSONAS, MemoryType.TOOLBOX, MemoryType.WORKFLOW_MEMORY, MemoryType.SUMMARIES
        ] else None
        
        # Retrieve using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                return collection.find_one({"_id": ObjectId(id)}, projection)
        except Exception:
            pass
            
        return None
    
    def retrieve_by_name(self, name: str, memory_store_type: MemoryType, include_embedding: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by name.
        
        Parameters:
        -----------
        name : str
            The name of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store to retrieve from.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}
        
        if memory_store_type == MemoryType.TOOLBOX:
            return self.toolbox_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.PERSONAS:
            return self.persona_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.workflow_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self.short_term_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            return self.long_term_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self.conversation_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.SUMMARIES:
            return self.summaries_collection.find_one({"name": name}, projection)
        


    def retrieve_persona_by_query(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a persona or several personas from MongoDB.
        This function uses a vector search to retrieve the most similar personas.

        Parameters:
        -----------
        query : Dict[str, Any]

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved personas, or None if not found.
        """

        # Get the embedding for the query
        embedding = get_embedding(query)

        # Create the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Execute the vector search
        results = list(self.persona_collection.aggregate(pipeline))

        # Return the results
        return results if results else None
        

    def retrieve_toolbox_item(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a toolbox item or several items from MongoDB.
        This function uses a vector search to retrieve the most similar toolbox items.
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of toolbox items to return.
        
        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved toolbox items, or None if not found.
        """

        # Get the embedding for the query
        embedding = get_embedding(query)

        # Create the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Execute the vector search
        results = list(self.toolbox_collection.aggregate(pipeline))

        # Return the results
        return results if results else None
    
    def retrieve_workflow_by_query(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve a workflow or several workflows from MongoDB.
        This function uses a vector search to retrieve the most similar workflows.

        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of workflows to return.
            
        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved workflows, or None if not found.
        """

        # Get the embedding for the query
        embedding = get_embedding(query)

        # Create the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Execute the vector search
        results = list(self.workflow_memory_collection.aggregate(pipeline))

        # Return the results
        return results if results else None

    def retrieve_summaries_by_query(self, query: Dict[str, Any], limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Retrieve summaries by query using vector search.
        
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of summaries to return.
            
        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved summaries, or None if not found.
        """
        # Get the embedding for the query
        embedding = get_embedding(query)

        # Create the vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit,
                    "index": "vector_index"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "embedding": 0,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]

        # Execute the vector search
        results = list(self.summaries_collection.aggregate(pipeline))

        # Return the results
        return results if results else None

    def get_summaries_by_memory_id(self, memory_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id, ordered by timestamp (most recent first).
        
        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        limit : int
            The maximum number of summaries to return.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries for the memory_id.
        """
        return list(self.summaries_collection.find(
            {"memory_id": memory_id}, 
            {"embedding": 0}
        ).sort("period_end", -1).limit(limit))

    def get_summaries_by_time_range(self, memory_id: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id within a time range based on the period they cover.
        
        NOTE: This filters by the time period that the summary covers (period_start/period_end),
        not when the summary was created. Use get_summaries_by_creation_time() to filter by creation time.
        
        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        start_time : float
            Start timestamp for the memory period range.
        end_time : float
            End timestamp for the memory period range.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries whose covered period falls within the time range.
        """
        from datetime import datetime
        
        # Convert to ISO string for compatibility with existing string timestamps
        start_iso = datetime.fromtimestamp(start_time).isoformat()
        end_iso = datetime.fromtimestamp(end_time).isoformat()
        
        # Query supports both float and string timestamps
        return list(self.summaries_collection.find({
            "memory_id": memory_id,
            "$or": [
                # Float timestamps (new format)
                {
                    "period_start": {"$gte": start_time, "$type": "number"},
                    "period_end": {"$lte": end_time, "$type": "number"}
                },
                # String timestamps (legacy format)
                {
                    "period_start": {"$gte": start_iso, "$type": "string"},
                    "period_end": {"$lte": end_iso, "$type": "string"}
                }
            ]
        }, {"embedding": 0}).sort("period_start", 1))

    def get_summaries_by_creation_time(self, memory_id: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id created within a time range.
        
        This filters by when the summary was actually created (created_at timestamp),
        not the time period that the summary covers.
        
        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        start_time : float
            Start timestamp for when summaries were created.
        end_time : float
            End timestamp for when summaries were created.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries created within the time range.
        """
        return list(self.summaries_collection.find({
            "memory_id": memory_id,
            "created_at": {"$gte": start_time, "$lte": end_time}
        }, {"embedding": 0}).sort("created_at", -1))

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by _id.
        
        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        # Get the appropriate collection
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.LONG_TERM_MEMORY: self.long_term_memory_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection
        }
        
        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            return False
            
        # Delete using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                result = collection.delete_one({"_id": ObjectId(id)})
                return result.deleted_count > 0
        except Exception:
            pass
            
        return False
    
    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by name.

        Parameters:
        -----------
        name : str
            The name of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.delete_one({"name": name}) 
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SUMMARIES:
            result = self.summaries_collection.delete_one({"name": name})
        else:
            return False
        
        return result.deleted_count > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """
        Delete all documents within a memory store type in MongoDB.
        
        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_many({})
        elif memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_many({})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            result = self.long_term_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.SUMMARIES:
            result = self.summaries_collection.delete_many({})
        else:
            return False
            
        return result.deleted_count > 0
            
    
    def list_all(self, memory_store_type: MemoryType, include_embedding: bool = False) -> List[Dict[str, Any]]:
        """
        List all documents within a memory store, decrypting fields if CSFLE is configured.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.
        
        Returns:
        --------
        List[Dict[str, Any]]
            The list of all documents from MongoDB.
        """
        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}

        collection = None
        if memory_store_type == MemoryType.PERSONAS:
            collection = self.persona_collection
        elif memory_store_type == MemoryType.TOOLBOX:
            collection = self.toolbox_collection
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            collection = self.short_term_memory_collection
        elif memory_store_type == MemoryType.LONG_TERM_MEMORY:
            collection = self.long_term_memory_collection
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            collection = self.conversation_memory_collection
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            collection = self.workflow_memory_collection
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            collection = self.shared_memory_collection
        elif memory_store_type == MemoryType.SUMMARIES:
            collection = self.summaries_collection
        else:
            logger.warning(f"Unsupported memory store type for list_all: {memory_store_type}")
            return []
        
        # Retrieve all documents from the collection
        documents = list(collection.find({}, projection))

        # --- Decryption Logic ---
        # Only attempt to decrypt if CSFLE is configured and initialized
        if self._client_encryption:
            decrypted_documents = []
            for doc in documents:
                decrypted_doc = doc.copy()
                for field_name, value in doc.items():
                    # Check if the value is a BSON Binary object (which indicates encryption)
                    if isinstance(value, Binary):
                        try:
                            # Attempt to decrypt the field
                            decrypted_value = self._client_encryption.decrypt(value)
                            decrypted_doc[field_name] = decrypted_value
                        except PyMongoError as e:
                            logger.error(f"Failed to decrypt field '{field_name}' in document {doc.get('_id', '')}: {e}")
                            # Keep the encrypted value if decryption fails
                            decrypted_doc[field_name] = value  
                decrypted_documents.append(decrypted_doc)
            return decrypted_documents

        # If CSFLE is not configured, just return the raw documents
        return documents

    def update_by_id(self, id: str, data: Dict[str, Any], memory_store_type: MemoryType) -> bool:
        """
        Update a document in a memory store type in MongoDB by _id.

        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to update.
        data : Dict[str, Any]
            The data to update the document with.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        
        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        # Get the appropriate collection
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.LONG_TERM_MEMORY: self.long_term_memory_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection
        }
        
        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            logger.error(f"No collection mapping found for memory store type: {memory_store_type}")
            return False
            
        # Update using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                result = collection.update_one({"_id": ObjectId(id)}, {"$set": data})
                success = result.modified_count > 0
                if not success:
                    logger.warning(f"Update operation found no documents to modify for id: {id}")
                return success
            else:
                logger.error(f"Invalid ObjectId: {id}")
                return False
        except Exception as e:
            logger.error(f"Error updating document with id {id}: {e}", exc_info=True)
            return False
            
            
    def update_toolbox_item(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a toolbox item in MongoDB by id using optimized queries.
        """

        # Update the embedding if the name, docstring or signature has changed

        # Get the old data
        old_data = self.retrieve_by_id(id, MemoryType.TOOLBOX)
        if not old_data:
            return False

        # Concatenate the name, docstring and signature if any of them have changed
        if old_data.get("name") != data.get("name"):
            data["name"] = data.get("name", old_data.get("name", ""))
        if old_data.get("docstring") != data.get("docstring"):
            data["docstring"] = data.get("docstring", old_data.get("docstring", ""))
        if old_data.get("signature") != data.get("signature"):
            data["signature"] = data.get("signature", old_data.get("signature", ""))

        # Update the embedding
        data["embedding"] = get_embedding(data["name"] + " " + data["docstring"] + " " + data["signature"])

        # Use the optimized update_by_id method
        return self.update_by_id(id, data, MemoryType.TOOLBOX)
    

    def retrieve_conversation_history_ordered_by_timestamp(self, memory_id: str, include_embedding: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to retrieve the conversation history for.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.

        Returns:
        --------
        List[Dict[str, Any]]
            The conversation history ordered by timestamp.
        """
        projection = {} if include_embedding else {"embedding": 0}
        return list(self.conversation_memory_collection.find({"memory_id": memory_id}, projection).sort("timestamp", 1))
    
    def retrieve_memory_units_by_query(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, memory_type: MemoryType = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memory units by query.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        memory_type : MemoryType
            The type of memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Detect the memory type
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return self.get_conversation_memory_units(query, query_embedding, memory_id, limit)
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return self.get_workflow_memory_units(query, query_embedding, memory_id, limit)
        elif memory_type == MemoryType.SUMMARIES:
            return self.get_summaries_memory_units(query, query_embedding, memory_id, limit)
        else:
            # Return empty list for unsupported memory types
            return []


    def get_conversation_memory_units(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the conversation memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Ensure vector index exists for conversation memory (lazy creation)
        if self.config.lazy_vector_indexes:
            self._ensure_vector_index_for_collection(
                self.conversation_memory_collection, 
                "conversation_memory", 
                memory_store=True
            )

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            query_embedding = get_embedding(query)

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": {"memory_id": memory_id}
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        # Execute the pipeline
        results = list(self.conversation_memory_collection.aggregate(pipeline))

        # Return the results
        return results

    def get_summaries_memory_units(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the summaries memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            query_embedding = get_embedding(query)

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": {"memory_id": memory_id}
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "created_at": -1}},
        ]

        # Execute the pipeline
        results = list(self.summaries_collection.aggregate(pipeline))

        # Return the results
        return results

    def get_workflow_memory_units(self, query: str = None, query_embedding: list[float] = None, memory_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the workflow memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Ensure vector index exists for workflow memory (lazy creation)
        if self.config.lazy_vector_indexes:
            self._ensure_vector_index_for_collection(
                self.workflow_memory_collection,
                "workflow_memory", 
                memory_store=True
            )

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            query_embedding = get_embedding(query)

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": {"memory_id": memory_id}
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        # Execute the pipeline
        results = list(self.workflow_memory_collection.aggregate(pipeline))

        # Return the results
        return results
    
    def store_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Store a memagent in the MongoDB database using only _id field.
        
        Parameters:
        -----------
        memagent : MemAgentModel
            The memagent to be stored.
        
        Returns:
        --------
        MemAgentModel
            The stored memagent.
        """
        # Convert the MemAgentModel to a dictionary
        memagent_dict = memagent.model_dump()
        
        # Remove agent_id field since we only want to use _id
        memagent_dict.pop("agent_id", None)
        
        # Convert persona to a serializable format if it exists
        if memagent.persona:
            # Store the entire persona object as a serializable dictionary
            memagent_dict["persona"] = memagent.persona.to_dict()
        
        # Remove any function objects from tools that could cause serialization issues
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]
        
        # Insert the document and let MongoDB generate _id automatically
        result = self.memagent_collection.insert_one(memagent_dict)
        
        # Add the generated _id to the response
        memagent_dict["_id"] = result.inserted_id

        return memagent_dict
    
    def update_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Update a memagent in the MongoDB database using _id field.
        """
        # Convert the MemAgentModel to a dictionary
        memagent_dict = memagent.model_dump()

        # Remove agent_id field since we only want to use _id
        agent_id = memagent_dict.pop("agent_id", None)
        
        # Convert persona to a serializable format if it exists
        if memagent.persona:
            memagent_dict["persona"] = memagent.persona.to_dict()
        
        # Remove any function objects from tools that could cause serialization issues
        if memagent_dict.get("tools") and isinstance(memagent_dict["tools"], list):
            for tool in memagent_dict["tools"]:
                if "function" in tool and callable(tool["function"]):
                    del tool["function"]
        
        # Update the memagent in the MongoDB database using _id
        if agent_id and ObjectId.is_valid(agent_id):
            self.memagent_collection.update_one(
                {"_id": ObjectId(agent_id)}, 
                {"$set": memagent_dict}
            )
        
        return memagent_dict

    
    def retrieve_memagent(self, agent_id: str) -> "MemAgentModel":
        """
        Retrieve a memagent from the MongoDB database using _id field.
        
        Parameters:
        -----------
        agent_id : str
            The agent ID to retrieve (MongoDB _id).
        
        Returns:
        --------
        MemAgentModel
            The retrieved memagent.
        """
        # Get the document from MongoDB using _id
        try:
            if ObjectId.is_valid(agent_id):
                document = self.memagent_collection.find_one({"_id": ObjectId(agent_id)}, {"embedding": 0})            
            else:
                return None
        except Exception:
            return None
        
        if not document:
            return None
        
        # Create a new MemAgent with data from the document
        # Use the MongoDB _id as agent_id since we no longer store agent_id field
        memagent = MemAgentModel(
            instruction=document.get("instruction"),
            application_mode=document.get("application_mode", "assistant"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=str(document.get("_id")),
            tools=document.get("tools"),
            long_term_memory_ids=document.get("long_term_memory_ids"),
            memory_provider=self
        )
        
        # Construct persona if present in the document
        if document.get("persona"):
            persona_data = document.get("persona")
            # Handle role as a string by matching it to a RoleType enum
            role_str = persona_data.get("role")
            role = None
            
            # Match the string role to a RoleType enum
            for role_type in RoleType:
                if role_type.value == role_str:
                    role = role_type
                    break
            
            # If no matching enum is found, default to GENERAL
            if role is None:
                role = RoleType.GENERAL
                
            memagent.persona = Persona(
                name=persona_data.get("name"),
                role=role,  # Pass the RoleType enum instead of string
                goals=persona_data.get("goals"),
                background=persona_data.get("background"),
                persona_id=persona_data.get("persona_id")
            )

        return memagent
    

    
    def list_memagents(self) -> List["MemAgentModel"]:
        """
        List all memagents in the MongoDB database.
        
        Returns:
        --------
        List[MemAgentModel]
            The list of memagents.
        """
        
        documents = list(self.memagent_collection.find({}, {"embedding": 0}))        
        agents = []
        
        for doc in documents:
            # Use the MongoDB _id as agent_id since we no longer store agent_id field
            agent = MemAgentModel(
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=str(doc.get("_id")),
                tools=doc.get("tools"),  # Include tools from document
                long_term_memory_ids=doc.get("long_term_memory_ids"),
                memory_provider=self
            )
            
            # Construct persona if present in the document
            if doc.get("persona"):
                persona_data = doc.get("persona")
                # Handle role as a string by matching it to a RoleType enum
                role_str = persona_data.get("role")
                role = None
                
                # Match the string role to a RoleType enum
                for role_type in RoleType:
                    if role_type.value == role_str:
                        role = role_type
                        break
                
                # If no matching enum is found, default to GENERAL
                if role is None:
                    role = RoleType.GENERAL
                    
                agent.persona = Persona(
                    name=persona_data.get("name"),
                    role=role,  # Pass the RoleType enum instead of string
                    goals=persona_data.get("goals"),
                    background=persona_data.get("background"),
                    persona_id=persona_data.get("persona_id")
                )
                
            agents.append(agent)
            
        return agents

    
    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """
        Update the memory_ids of a memagent in the memory provider using _id field.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update (MongoDB _id).
        memory_ids : List[str]
            The list of memory_ids to update.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        try:
            if ObjectId.is_valid(agent_id):
                result = self.memagent_collection.update_one(
                    {"_id": ObjectId(agent_id)}, 
                    {"$set": {"memory_ids": memory_ids}}
                )
                return result.modified_count > 0
            else:
                return False
        except Exception:
            return False
    
    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """
        Delete the memory_ids of a memagent in the memory provider.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update (MongoDB _id).

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        try:
            if ObjectId.is_valid(agent_id):
                result = self.memagent_collection.update_one(
                    {"_id": ObjectId(agent_id)}, 
                    {"$unset": {"memory_ids": []}}
                )
                return result.modified_count > 0
            else:
                return False
        except Exception:
            return False
    
    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """
        Delete a memagent from the memory provider by id.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to delete.
        cascade : bool
            Whether to cascade the deletion of the memagent. This deletes all the memory units associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if cascade:
            # Retrieve the memagent
            memagent = self.retrieve_memagent(agent_id)

            if memagent is None:
                raise ValueError(f"MemAgent with id {agent_id} not found")
            
            # Delete all the memory units associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.
            for memory_id in memagent.memory_ids:
                # Loop through all the memory stores and delete records with the memory_ids
                for memory_type in MemoryType:
                    self._delete_memory_units_by_memory_id(memory_id, memory_type)
        else:
            try:
                if ObjectId.is_valid(agent_id):
                    result = self.memagent_collection.delete_one({"_id": ObjectId(agent_id)})
                    return result.deleted_count > 0
                else:
                    return False
            except Exception:
                return False

        return True
    
    def _delete_memory_units_by_memory_id(self, memory_id: str, memory_type: MemoryType):
        """
        Delete all the memory units associated with the memory_id.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to delete.
        memory_type : MemoryType
            The type of memory to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            self.conversation_memory_collection.delete_many({"memory_id": memory_id})

        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            self.workflow_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.SHORT_TERM_MEMORY:
            self.short_term_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.LONG_TERM_MEMORY:
            self.long_term_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.PERSONAS:
            self.persona_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.TOOLBOX:
            self.toolbox_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.MEMAGENT:
            self.memagent_collection.delete_many({"memory_id": memory_id})
                    

    def _setup_vector_search_index(self, collection, index_name="vector_index", memory_store: bool = False):
        """
        Setup a vector search index for a MongoDB collection and wait for it to become queryable.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """

        # Define the index definition
        vector_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    # Dynamic dimensions based on the configured embedding provider
                    "numDimensions": self._get_embedding_dimensions_safe(),
                    "similarity": "cosine",
                }
            ]
        }

        # If the memory store is true, then we add the memory_id field to the index
        # This is used to prefilter the memory units by memory_id
        # useful to narrow the scope of your semantic search and ensure that not all vectors are considered for comparison. 
        # It reduces the number of documents against which to run similarity comparisons, which can decrease query latency and increase the accuracy of search results.
        if memory_store:
            vector_index_definition["fields"].append({
                "type": "filter",
                "path": "memory_id",
            })

        new_vector_search_index_model = SearchIndexModel(
            definition=vector_index_definition, name=index_name, type="vectorSearch"
        )

        # Create the new index
        try:
            result = collection.create_search_index(model=new_vector_search_index_model)
            print(f"Creating index '{index_name}'... for collection {collection.name}")
            
            # Wait for the index to become queryable using polling mechanism
            self._wait_for_index_ready(collection, result, index_name)
            
            return result

        except Exception as e:
            print(f"Error creating new vector search index '{index_name}': {e!s}")
            return None

    def _wait_for_index_ready(self, collection, index_name_result, display_name="vector_index"):
        """
        Wait for a MongoDB Atlas search index to become queryable using polling.
        
        Args:
        collection: MongoDB collection object
        index_name_result: The name/result returned from create_search_index
        display_name: Human-readable name for logging (default: "vector_index")
        """
        print(f"Polling to check if the index '{index_name_result}' is ready. This may take up to a minute.")
        
        # Define predicate function to check if index is queryable
        predicate = lambda index: index.get("queryable") is True
        
        while True:
            try:
                # List search indexes and find the one we just created
                indices = list(collection.list_search_indexes(index_name_result))
                
                # Check if the index exists and is queryable
                if indices and predicate(indices[0]):
                    break
                    
                # Wait 5 seconds before checking again
                time.sleep(5)
                
            except Exception as e:
                print(f"Error checking index readiness: {e}")
                # Continue polling even if there's an error
                time.sleep(5)
        
        print(f"Index '{index_name_result}' is ready for querying in collection {collection.name}.")
        
    def _ensure_vector_index(self, collection, index_name="vector_index", memory_store: bool = False):
        """
        Ensure a vector search index exists for the collection. If it doesn't exist, create it and wait for it to be ready.
        
        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """
        search_indexes = list(collection.list_search_indexes())
        has_vector_index = any(index.get("name") == index_name and index.get("type") == "vectorSearch" for index in search_indexes)
        
        if not has_vector_index:
            print(f"Vector search index '{index_name}' not found for collection {collection.name}. Creating...")
            self._setup_vector_search_index(collection, index_name, memory_store)
            print(f"Vector index '{index_name}' for {collection.name} collection is now ready for use.")
        else:
            print(f"Vector search index '{index_name}' already exists for collection {collection.name}.")

    def close(self) -> None:
        """Close the connection to MongoDB."""
        self.client.close()