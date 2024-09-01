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

First, import the necessary components and set up your environment:

```python
from memorizz.database.mongodb import mongodb_tools, MongoDBToolsConfig, MongoDBTools, get_mongodb_toolbox
import os
import getpass

# Set up your OpenAI API key and MongoDB URI
OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

MONGO_URI = getpass.getpass("Enter MongoDB URI: ")
os.environ["MONGO_URI"] = MONGO_URI
```

#### Creating a MongoDBTools Instance
You can create a MongoDBTools instance in two ways:

```python
# Option 1: Create MongoDBTools instance directly
config = MongoDBToolsConfig(mongo_uri=MONGO_URI)
mongodb_tools = MongoDBTools(config)

# Option 2: Use the get_mongodb_toolbox function
mongodb_toolbox = get_mongodb_toolbox(config)
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
user_query = "Hi, can you shout the statement: We are there"
tools = mongodb_tools.populate_tools(user_query)
```

The populate_tools method performs a vector search in the MongoDB database to find the most relevant tools based on the user's query.

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
