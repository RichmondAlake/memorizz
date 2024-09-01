MemoRizz is a comprehensive library for AI-assisted tools and functionalities. It currently includes MongoDB tools for storing function definitions and performing customizable vector searches.

## Installation

```
pip install memorizz
```

## Usage

MemoRizz can be used with either a custom configuration or the default settings. Here are examples of both approaches:

### Using Custom Configuration

```python
from memorizz.database.mongodb import MongoDBTools, MongoDBToolsConfig, mongodb_toolbox

# Create a custom configuration
custom_config = MongoDBToolsConfig(
    mongo_uri="mongodb://custom-uri:27017",
    db_name="my_custom_db",
    collection_name="my_custom_tools",
    embedding_model="text-embedding-ada-002",
    vector_search_candidates=200,
    vector_index_name="my_custom_index"
)

# Initialize MongoDBTools with custom configuration
custom_mongo_tools = MongoDBTools(custom_config)

# Use the custom mongodb_toolbox decorator
@custom_mongo_tools.mongodb_toolbox()
def custom_function(param1: str, param2: int) -> str:
    """
    This is a custom function using a custom configuration.
    """
    return f"Custom processed {param1} with {param2}"

# Use the custom tools
user_query = "How do I process something custom?"
custom_tools = custom_mongo_tools.populate_tools(user_query, num_tools=3)

print("Custom tools:", custom_tools)
```

### Using Default Configuration

```python
from memorizz.databases.mongodb import mongodb_toolbox, MongoDBTools

# Use the default mongodb_toolbox decorator
@mongodb_toolbox()
def default_function(param1: str, param2: int) -> str:
    """
    This is a default function using the default configuration.
    """
    return f"Default processed {param1} with {param2}"

# Use the default tools
default_mongo_tools = MongoDBTools()  # This uses the default configuration
user_query = "How do I process something with default settings?"
default_tools = default_mongo_tools.populate_tools(user_query, num_tools=2)

print("Default tools:", default_tools)
```

In both cases, the `mongodb_toolbox` decorator is used to register functions, and the `populate_tools` method is used to retrieve relevant tools based on a user query. The main difference is in the setup and configuration.

The custom configuration approach allows you to specify your own MongoDB URI, database name, collection name, and other settings. This is useful when you need to connect to a specific MongoDB instance or customize the behavior of the tools.

The default configuration approach is simpler and requires less setup. It's suitable for quick starts or when the default settings meet your needs.

Choose the approach that best fits your project requirements and structure.

## Features

- MongoDB Tools:
  - Automatic tool registration with MongoDB
  - Vector search based on user queries
  - Customizable number of tools returned
  - Configurable embedding model and search parameters
  - Error handling and logging
- Extensible structure for future database support


## Feature Roadmap

- [x] Implement basic MongoDB tools functionality
- [x] Add customizable vector search
- [x] Implement error handling and logging
- [x] Create PyPI package
- [x] Restructure package for extensibility to other databases
- [ ] Add support for multiple embedding models to make MongoDBToolsConfig embedding model agnostic
- [ ] Implement MemScore logic for improved tool ranking and memory component retrival
- [ ] Add async support for improved performance
- [ ] Implement caching mechanism for embeddings and search results
- [ ] Create comprehensive API documentation
- [ ] Implement automated testing suite with high coverage
- [ ] Create example projects and use cases
- [ ] Add support for additional databases (e.g., PostgreSQL, Redis)

## License

This project is licensed under the MIT License.
