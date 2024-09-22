import os
import getpass
import inspect
from functools import wraps
from typing import get_type_hints, List, Dict, Any, Optional
import openai
import pymongo
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MongoDBToolsConfig:
    mongo_uri: Optional[str] = None
    db_name: str = 'function_calling_db'
    collection_name: str = 'tools'
    embedding_model: str = "text-embedding-3-small",
    embeding_dimensions_size: int = 256,
    vector_search_candidates: int = 150,
    vector_index_name: str = "vector_index"

def get_embedding(text: str, model: str = "text-embedding-3-small", dimensions: int = 256) -> List[float]:
    text = text.replace("\n", " ")
    try:
        return openai.OpenAI().embeddings.create(input=[text], model=model, dimensions=dimensions).data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

class MongoDBTools:
    def __init__(self, config: MongoDBToolsConfig = MongoDBToolsConfig()):
        self.config = config
        if self.config.mongo_uri is None:
            self.config.mongo_uri = os.getenv('MONGO_URI') or getpass.getpass("Enter MongoDB URI: ")
        
        self.mongo_client = None
        self.db = None
        self.tools_collection = None
        
        try:
            self.mongo_client = pymongo.MongoClient(self.config.mongo_uri, appname="memorizz.python.package")
            self.db = self.mongo_client[self.config.db_name]

            # Check if collection exists, create if it doesn't
            if self.config.collection_name not in self.db.list_collection_names():
                self.db.create_collection(self.config.collection_name)
                logger.info(f"Collection '{self.config.collection_name}' created.")
            
            self.tools_collection = self.db[self.config.collection_name]

            # Check if vector search index exists, create if it doesn't
            self._ensure_vector_search_index()

            logger.info("MongoDBTools initialized successfully.")

        except pymongo.errors.ConnectionFailure:
            logger.error("Failed to connect to MongoDB. Please check your connection string and network.")
        except pymongo.errors.OperationFailure as e:
            if e.code == 13:  # Authentication failed
                logger.error("MongoDB authentication failed. Please check your credentials.")
            else:
                logger.error(f"MongoDB operation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB initialization: {str(e)}")
        
        if self.tools_collection is None:
            logger.warning("MongoDBTools initialization failed. Some features may not work.")

    def mongodb_toolbox(self, collection: Optional[Collection] = None):
        if collection is None:
            collection = self.tools_collection

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""

            if not docstring:
                raise ValueError(f"Error registering tool {func.__name__}: Docstring is missing. Please provide a docstring for the function.")

            type_hints = get_type_hints(func)

            tool_def = {
                "name": func.__name__,
                "description": docstring.strip(),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            for param_name, param in signature.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                param_type = type_hints.get(param_name, type(None))
                json_type = "string"
                if param_type in (int, float):
                    json_type = "number"
                elif param_type == bool:
                    json_type = "boolean"

                tool_def["parameters"]["properties"][param_name] = {
                    "type": json_type,
                    "description": f"Parameter {param_name}"
                }

                if param.default == inspect.Parameter.empty:
                    tool_def["parameters"]["required"].append(param_name)

            tool_def["parameters"]["additionalProperties"] = False

            try:
                vector = get_embedding(tool_def["description"], self.config.embedding_model, self.config.embeding_dimensions_size)
                tool_doc = {
                    **tool_def,
                    "embedding": vector
                }
                collection.update_one({"name": func.__name__}, {"$set": tool_doc}, upsert=True)
                logger.info(f"Successfully registered tool: {func.__name__}")
            except Exception as e:
                logger.error(f"Error registering tool {func.__name__}: {str(e)}")
                raise

            return wrapper
        return decorator

    def _vector_search(self, user_query: str, collection: Optional[Collection] = None, limit: int = 2) -> List[Dict[str, Any]]:
        if collection is None:
            collection = self.tools_collection

        try:
            query_embedding = get_embedding(user_query, self.config.embedding_model, self.config.embeding_dimensions_size)
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            raise

        vector_search_stage = {
            "$vectorSearch": {
                "index": self.config.vector_index_name,
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": self.config.vector_search_candidates,
                "limit": limit
            }
        }

        unset_stage = {
            "$unset": "embedding"
        }

        pipeline = [vector_search_stage, unset_stage]

        try:
            results = collection.aggregate(pipeline)
            return list(results)
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise

    def populate_tools(self, user_query: str, num_tools: int = 2) -> List[Dict[str, Any]]:
        try:
            search_results = self._vector_search(user_query, limit=num_tools)
            tools = []
            for result in search_results:
                print(result)
                tool = {
                    "type": "function",
                    "function": {
                        "name": result["name"],
                        "description": result["description"],
                        "parameters": result["parameters"]
                    }
                }
                tools.append(tool)
            logger.info(f"Successfully populated {len(tools)} tools")
            return tools
        except Exception as e:
            logger.error(f"Error populating tools: {str(e)}")
            raise

    def create_toolbox(self, collection_name: str, vector_index_definition: Dict[str, Any], index_name: str = "vector_index"):
        """
        Create a collection in the MongoDB database and set up a vector search index.

        Args:
        collection_name (str): Name of the collection to create
        vector_index_definition (Dict[str, Any]): Dictionary containing the vector index definition
        index_name (str): Name of the index (default: "vector_index")

        Returns:
        bool: True if the toolbox was created successfully, False otherwise
        """
        try:
            # Create the collection
            collection = self.db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created successfully.")

            # Create the vector search index
            search_index_model = SearchIndexModel(
                definition=vector_index_definition,
                name=index_name
            )

            result = collection.create_search_index(model=search_index_model)
            logger.info(f"Vector search index '{index_name}' created successfully for collection '{collection_name}'.")

            # Update the config to use the new collection
            self.config.collection_name = collection_name
            self.config.vector_index_name = index_name
            self.tools_collection = collection

            return True

        except pymongo.errors.CollectionInvalid:
            logger.warning(f"Collection '{collection_name}' already exists. Using existing collection.")
            collection = self.db[collection_name]
            self.tools_collection = collection
            
            # Check if the index already exists
            existing_indexes = collection.list_indexes()
            index_exists = any(index['name'] == index_name for index in existing_indexes)
            
            if not index_exists:
                # Create the index if it doesn't exist
                search_index_model = SearchIndexModel(
                    definition=vector_index_definition,
                    name=index_name
                )
                result = collection.create_search_index(model=search_index_model)
                logger.info(f"Vector search index '{index_name}' created successfully for existing collection '{collection_name}'.")
            else:
                logger.info(f"Vector search index '{index_name}' already exists for collection '{collection_name}'.")

            # Update the config to use the existing collection
            self.config.collection_name = collection_name
            self.config.vector_index_name = index_name

            return True

        except Exception as e:
            logger.error(f"Error creating toolbox: {str(e)}")
            return False

    def _ensure_vector_search_index(self):
        try:
            indexes = list(self.tools_collection.list_indexes())
            index_exists = any(index['name'] == self.config.vector_index_name for index in indexes)
            
            if not index_exists:
                vector_index_definition = {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "dimensions": self.config.embeding_dimensions_size,
                                "similarity": "cosine",
                                "type": "knnVector",
                            }
                        }
                    }
                }
                try:
                    # Create SearchIndexModel and use it in create_search_index
                    search_index_model = SearchIndexModel(
                        definition=vector_index_definition,
                        name=self.config.vector_index_name
                    )
                    self.tools_collection.create_search_index(search_index_model)
                    logger.info(f"Vector search index '{self.config.vector_index_name}' created.")
                except pymongo.errors.OperationFailure as e:
                    if e.code == 68 or "already exists" in str(e):
                        logger.info(f"Vector search index '{self.config.vector_index_name}' already exists.")
                    else:
                        logger.warning(f"Unexpected error while creating index: {str(e)}")
            else:
                logger.info(f"Vector search index '{self.config.vector_index_name}' already exists.")
        except Exception as e:
            logger.warning(f"Note on vector search index: {str(e)}")
            
__all__ = ['MongoDBTools', 'MongoDBToolsConfig', 'get_embedding']

# You can create a function to get the mongodb_toolbox decorator:
def get_mongodb_toolbox(config: MongoDBToolsConfig = MongoDBToolsConfig()):
    return MongoDBTools(config).mongodb_toolbox
