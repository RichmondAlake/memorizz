# MemoRizz

[![PyPI version](https://badge.fury.io/py/memorizz.svg)](https://badge.fury.io/py/memorizz)
[![PyPI downloads](https://img.shields.io/pypi/dm/memorizz.svg)](https://pypistats.org/packages/memorizz)


`memorizz` is a Python library that seamlessly integrates MongoDB with OpenAI's embedding capabilities to create a dynamic function toolbox. It allows developers to easily store, retrieve, and manage function definitions as tools, making it ideal for building AI assistants, chatbots, and other natural language processing applications.

## Key Features

- **MongoDB Integration**: Efficiently store and manage function definitions in MongoDB.
- **OpenAI Embeddings**: Utilize OpenAI's powerful embedding models for semantic search capabilities.
- **Dynamic Tool Retrieval**: Retrieve relevant function tools based on natural language queries.
- **Decorator-based Registration**: Easily register Python functions as tools using a simple decorator.
- **Flexible Configuration**: Customize MongoDB and embedding settings to fit your project needs.
- **AI-Ready Output**: Generate tool descriptions compatible with OpenAI's function calling format.

Whether you're building a complex AI system or simply want to organize and retrieve functions semantically, `memorizz` provides a robust foundation for your project. It bridges the gap between your code and natural language interfaces, enabling more intelligent and context-aware applications.

## Installation

```
pip install memorizz
```

## Vector Database as A Toolbox
Use a vector database as a storage solution for tools within AI agents and agentic systems.

Function calling in AI models, while powerful, comes with several limitations:

1. Token Usage: Functions count against the model's context limit and are billed as input tokens, potentially leading to high costs for complex applications.
2. Accuracy with Multiple Functions: Model accuracy typically decreases when choosing between 10-20+ functions, limiting the scalability of function-based approaches.
3. Context Limitations: Providing adequate context for function usage often relies heavily on carefully crafted system messages.

Current Solutions
To address these limitations, developers often resort to:

- Carefully limiting the number of functions exposed to the model
- Extensive prompt engineering to guide function selection
- Model fine-tuning for specific function sets
- Building complex multi-agent systems to manage large function libraries

### How Memorizz Toolbox Functionality Helps
Memorizz introduces a dynamic, scalable approach to function management that addresses these limitations:

1. Efficient Token Usage: By storing function definitions in MongoDB and using vector search, memorizz reduces the need to send all function definitions with each API call, potentially saving on token usage.
2. Improved Accuracy at Scale: The vector search capability allows memorizz to present only the most relevant functions to the model, maintaining accuracy even with large function libraries.
3. Semantic Search: Vector-based search allows finding relevant functions even when queries don't exactly match function names or descriptions.
4. Scalability: Database storage allows for managing thousands of functions efficiently, far beyond the practical limits of traditional function calling.



### Using MongoDB as a Toolbox

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RichmondAlake/memorizz/blob/main/test.ipynb)


The `memorizz` library provides functionality to use MongoDB as a toolbox for storing and retrieving function definitions. Here's how to use this feature:

### Setup

To use MongoDB as a toolbox, you will need to complete the following steps:

1. Register for a MongoDB Account:
   - Go to the MongoDB website (https://www.mongodb.com/cloud/atlas/register).
   - Click on the "Try Free" or "Get Started Free" button.
   - Fill out the registration form with your details and create an account.

2. Create a [MongoDB Cluster](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/#procedure)

3. Set Up [Database Access](https://www.mongodb.com/docs/atlas/security-add-mongodb-users/#add-database-users):
   - In the left sidebar, click on "Database Access" under "Security".
   - Click "Add New Database User".
   - Create a username and a strong password. Save these credentials securely.
   - Set the appropriate permissions for the user (e.g., "Read and write to any database").

4. Configure Network Access:
   - In the left sidebar, click on "Network Access" under "Security".
   - Click "Add IP Address".
   - To allow access from anywhere (not recommended for production), enter 0.0.0.0/0.
   - For better security, whitelist only the specific IP addresses that need access.

5. Obtain the [MongoDB URI](https://www.mongodb.com/docs/manual/reference/connection-string/#find-your-mongodb-atlas-connection-string):

Now that you have your MongoDB URI, you can use it to connect to your cluster in the `memorizz` library.


First, import the necessary components and set up your environment:

```python
from memorizz.database.mongodb import MongoDBToolsConfig, MongoDBTools
import os
import getpass

# Set up environment variables for API keys and MongoDB URI
# This key is required for using OpenAI's services, such as generating embeddings
OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# This URI is needed to connect to the MongoDB database
MONGO_URI = getpass.getpass("Enter MongoDB URI: ")
os.environ["MONGO_URI"] = MONGO_URI
```

#### Creating a MongoDBTools Instance
You can create a MongoDBTools instance in two ways:

```python
# 1. Initialize the MongoDB configuration and create a MongoDB tools instance
config = MongoDBToolsConfig(
    mongo_uri=MONGO_URI,  # MongoDB connection string
    db_name="function_calling_db",  # Name of the database to use
    collection_name="tools",  # Name of the collection to store tools
    embedding_model="text-embedding-3-small",  # OpenAI model for generating embeddings
    embeding_dimensions_size=256,  # Dimension size for the embedding vectors
    vector_search_candidates=150,  # Number of candidates to consider in vector search
    vector_index_name="vector_index"  # Name of the vector index in MongoDB
)

# 2. Create an instance of MongoDBTools with the configured settings
mongodb_tools = MongoDBTools(config)

# 3. Create a decorator function for registering tools
mongodb_toolbox = mongodb_tools.mongodb_toolbox
```

#### Registering Functions as Tools

Use the `@mongodb_toolbox` decorator to register functions as tools. It's crucial to include a detailed Google-style docstring for each function, as this docstring is what gets embedded and utilized for semantic retrieval.

The docstring should include:

- A brief description of what the function does
- When to use the function in the context of user queries
- Detailed descriptions of parameters
- Information about the return value
- An example of how to use the function

```python
import random
from datetime import datetime

# 4. Define and register tool functions using the mongodb_toolbox decorator
# These functions will be stored in the MongoDB database and can be retrieved for function calling
@mongodb_toolbox()
def shout(statement: str) -> str:
  """
  Convert a statement to uppercase letters to emulate shouting. Use this when a user wants to emphasize something strongly or when they explicitly ask to 'shout' something..

  """
  return statement.upper()

@mongodb_toolbox()
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a specified location.
    Use this when a user asks about the weather in a specific place.

    :param location: The name of the city or location to get weather for.
    :param unit: The temperature unit, either 'celsius' or 'fahrenheit'. Defaults to 'celsius'.
    :return: A string describing the current weather.
    """
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temperature = random.randint(-10, 35)

    if unit.lower() == "fahrenheit":
        temperature = (temperature * 9/5) + 32

    condition = random.choice(conditions)
    return f"The weather in {location} is currently {condition} with a temperature of {temperature}°{'C' if unit.lower() == 'celsius' else 'F'}."

@mongodb_toolbox()
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a given stock symbol.
    Use this when a user asks about the current price of a specific stock.

    :param symbol: The stock symbol to look up (e.g., 'AAPL' for Apple Inc.).
    :return: A string with the current stock price.
    """
    price = round(random.uniform(10, 1000), 2)
    return f"The current stock price of {symbol} is ${price}."

@mongodb_toolbox()
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time for a specified timezone.
    Use this when a user asks about the current time in a specific timezone.

    :param timezone: The timezone to get the current time for. Defaults to 'UTC'.
    :return: A string with the current time in the specified timezone.
    """
    current_time = datetime.utcnow().strftime("%H:%M:%S")
    return f"The current time in {timezone} is {current_time}."

```

#### Retrieving Tools Based on User Query
You can retrieve relevant tools based on a user query:

```python
# 5. Define the user query
# This query will be used to search for relevant tools in the MongoDB database
user_query = "Hi, can you shout the statement: We are there"

# 6. Populate tools based on the user query
tools = mongodb_tools.populate_tools(
    user_query,  # The query string to search for relevant tools
    num_tools=2  # The maximum number of tools to return from the search
    )
```
The populate_tools method performs a vector search in the MongoDB database to find the most relevant tools based on the user's query.

```python
import pprint

pprint.pprint(tools)
```

```json
[{'function': {'description': 'Convert a statement to uppercase letters to '
                              'emulate shouting. Use this when a user wants to '
                              'emphasize something strongly or when they '
                              "explicitly ask to 'shout' something..",
               'name': 'shout',
               'parameters': {'additionalProperties': False,
                              'properties': {'statement': {'description': 'Parameter '
                                                                          'statement',
                                                           'type': 'string'}},
                              'required': ['statement'],
                              'type': 'object'}},
  'type': 'function'},
 {'function': {'description': 'Get the current stock price for a given stock '
                              'symbol.\n'
                              'Use this when a user asks about the current '
                              'price of a specific stock.\n'
                              '\n'
                              ':param symbol: The stock symbol to look up '
                              "(e.g., 'AAPL' for Apple Inc.).\n"
                              ':return: A string with the current stock price.',
               'name': 'get_stock_price',
               'parameters': {'additionalProperties': False,
                              'properties': {'symbol': {'description': 'Parameter '
                                                                       'symbol',
                                                        'type': 'string'}},
                              'required': ['symbol'],
                              'type': 'object'}},
  'type': 'function'}]
```

#### Using the Retrieved Tools

The retrieved tools can be used with language models or other applications. Each tool is represented as a dictionary with the following structure:

```json
{
    'type': 'function',
    'function': {
        'name': 'function_name',
        'description': 'function_description',
        'parameters': {
            'type': 'object',
            'properties': {
                'param1': {'description': 'Parameter description', 'type': 'string'},
                # ... other parameters ...
            },
            'required': ['param1'],
            'additionalProperties': False
        }
    }
}

```

You can iterate through the tools and use them as needed in your application.
#### Note
Make sure you have set up your MongoDB instance and have the necessary permissions to read and write to the specified database and collection. Also, ensure that your OpenAI API key is valid and has the required permissions for generating embeddings.

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
