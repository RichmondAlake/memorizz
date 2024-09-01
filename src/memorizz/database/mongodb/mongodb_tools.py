import os
import json
import getpass
import inspect
from functools import wraps
from typing import get_type_hints, List, Dict, Any, Optional
import openai
import pymongo
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MongoDBToolsConfig:
    mongo_uri: Optional[str] = None
    db_name: str = 'function_calling_db'
    collection_name: str = 'tools'
    embedding_model: str = "text-embedding-3-small"
    vector_search_candidates: int = 150
    vector_index_name: str = "vector_index"

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    try:
        return openai.OpenAI().embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

class MongoDBTools:
    def __init__(self, config: MongoDBToolsConfig = MongoDBToolsConfig()):
        self.config = config
        if self.config.mongo_uri is None:
            self.config.mongo_uri = os.getenv('MONGO_URI') or getpass.getpass("Enter MongoDB URI: ")
        try:
            self.mongo_client = pymongo.MongoClient(self.config.mongo_uri)
            self.db = self.mongo_client[self.config.db_name]
            self.tools_collection = self.db[self.config.collection_name]
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def mongodb_toolbox(self, collection: Optional[pymongo.collection.Collection] = None):
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
                vector = get_embedding(tool_def["description"], self.config.embedding_model)
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

    def _vector_search(self, user_query: str, collection: Optional[pymongo.collection.Collection] = None, limit: int = 2) -> List[Dict[str, Any]]:
        if collection is None:
            collection = self.tools_collection

        try:
            query_embedding = get_embedding(user_query, self.config.embedding_model)
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

__all__ = ['MongoDBTools', 'MongoDBToolsConfig', 'get_embedding']

# You can create a function to get the mongodb_toolbox decorator:
def get_mongodb_toolbox(config: MongoDBToolsConfig = MongoDBToolsConfig()):
    return MongoDBTools(config).mongodb_toolbox
