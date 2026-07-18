# Embedding Provider Configuration

Memorizz provides a flexible and scalable embedding configuration system that allows you to choose between different embedding providers and configure them according to your needs.

## Folder Structure

The embedding providers are organized in a clean, scalable folder structure:

```
src/memorizz/embeddings/
├── __init__.py          # Main interfaces, manager, and configuration
├── README.md            # This documentation file
├── openai/              # OpenAI provider package
│   ├── __init__.py      # Package exports
│   └── provider.py      # OpenAI embedding implementation
├── ollama/              # Ollama provider package
│   ├── __init__.py      # Package exports
│   └── provider.py      # Ollama embedding implementation
└── voyageai/            # VoyageAI provider package
    ├── __init__.py      # Package exports
    └── provider.py      # VoyageAI embedding implementation (text, multimodal, contextualized)
```

This organization makes it easy to:
- **Add new providers**: Simply create a new folder with `__init__.py` and `provider.py`
- **Maintain providers**: Each provider's code is self-contained
- **Scale the system**: No limit on the number of embedding providers
- **Import cleanly**: `from memorizz.embeddings.openai import OpenAIEmbeddingProvider`

## Supported Providers

### OpenAI Embeddings
- **Models**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Configurable dimensions**: Yes (for 3-small and 3-large models)
- **Quality**: High
- **Privacy**: Requires API calls to OpenAI
- **Cost**: Pay per embedding

### Ollama Embeddings
- **Models**: `nomic-embed-text`, `mxbai-embed-large`, `snowflake-arctic-embed`, `all-minilm`
- **Configurable dimensions**: Fixed per model
- **Quality**: Good (varies by model)
- **Privacy**: Complete local processing
- **Cost**: Free (local inference)

### VoyageAI Embeddings
- **Models**: `voyage-3.5`, `voyage-3-large`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2`, `voyage-multimodal-3`, `voyage-context-3`
- **Configurable dimensions**: Yes (varies by model, supports 256-2048 dimensions)
- **Quality**: Very high with specialized models for different domains
- **Privacy**: Requires API calls to VoyageAI
- **Cost**: Pay per embedding
- **Special Features**: Multimodal (text+images) and contextualized chunk embeddings

## Quick Start

### Global Configuration

Configure embeddings globally for your entire application:

```python
from memorizz.embeddings import configure_embeddings

# Configure OpenAI embeddings
configure_embeddings(
    provider="openai",
    config={
        "model": "text-embedding-3-small",
        "dimensions": 512,
        "api_key": "your-api-key"  # Optional if using env var
    }
)

# Or configure Ollama embeddings
configure_embeddings(
    provider="ollama",
    config={
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434"
    }
)

# Or configure VoyageAI embeddings
configure_embeddings(
    provider="voyageai",
    config={
        "embedding_type": "text",
        "model": "voyage-3.5",
        "output_dimension": 512,
        "api_key": "your-voyage-api-key"  # Optional if using env var
    }
)
```

### Per-Agent Configuration

Configure embeddings for specific agents:

```python
from memorizz.memagent import MemAgent

# Agent with OpenAI embeddings
agent = MemAgent(
    instruction="You are a helpful assistant.",
    embedding_provider="openai",
    embedding_config={
        "model": "text-embedding-3-large",
        "dimensions": 1024
    }
)

# Agent with Ollama embeddings
agent = MemAgent(
    instruction="You are a privacy-focused assistant.",
    embedding_provider="ollama",
    embedding_config={
        "model": "nomic-embed-text"
    }
)

# Agent with VoyageAI embeddings
agent = MemAgent(
    instruction="You are a high-quality assistant.",
    embedding_provider="voyageai",
    embedding_config={
        "embedding_type": "text",
        "model": "voyage-3.5",
        "output_dimension": 1024,
        "input_type": "query"
    }
)
```

## Detailed Configuration

### OpenAI Configuration Options

```python
openai_config = {
    "model": "text-embedding-3-small",  # Model to use
    "dimensions": 256,                  # Embedding dimensions (128-1536 for 3-small)
    "api_key": "your-key",             # Optional API key
    "base_url": "https://api.openai.com/v1"  # Optional custom endpoint
}
```

**Available Models:**
- `text-embedding-3-small`: Up to 1536 dimensions, efficient
- `text-embedding-3-large`: Up to 3072 dimensions, highest quality
- `text-embedding-ada-002`: Fixed 1536 dimensions, legacy model

### Ollama Configuration Options

```python
ollama_config = {
    "model": "nomic-embed-text",        # Model to use
    "base_url": "http://localhost:11434", # Ollama server URL
    "timeout": 30                       # Request timeout in seconds
}
```

**Available Models:**
- `nomic-embed-text`: 768 dimensions, general purpose
- `mxbai-embed-large`: 1024 dimensions, high quality
- `snowflake-arctic-embed`: 1024 dimensions, good performance
- `all-minilm`: 384 dimensions, compact and fast

### VoyageAI Configuration Options

```python
voyageai_config = {
    "embedding_type": "text",               # Type: "text", "multimodal", "contextualized"
    "model": "voyage-3.5",                  # Model to use
    "output_dimension": 512,                # Embedding dimensions (varies by model)
    "output_dtype": "float",                # Data type: "float", "int8", "uint8", "binary", "ubinary"
    "input_type": "document",               # Optional: "query", "document", None
    "api_key": "your-key"                   # Optional API key
}
```

**Available Models:**

*Text Models:*
- `voyage-3.5`: 1024 dimensions (default), supports 256/512/1024/2048
- `voyage-3-large`: 1024 dimensions (default), supports 256/512/1024/2048
- `voyage-3.5-lite`: 1024 dimensions (default), supports 256/512/1024/2048
- `voyage-code-3`: 1024 dimensions (default), supports 256/512/1024/2048
- `voyage-finance-2`: 1024 dimensions (fixed)
- `voyage-law-2`: 1024 dimensions (fixed)
- `voyage-code-2`: 1536 dimensions (fixed)

*Multimodal Models:*
- `voyage-multimodal-3`: 1024 dimensions, text + images

*Contextualized Models:*
- `voyage-context-3`: 1024 dimensions (default), supports 256/512/1024/2048

## Advanced Usage

### Multiple Agents with Different Providers

```python
# High-precision agent with OpenAI
research_agent = MemAgent(
    instruction="Research assistant requiring high precision.",
    embedding_provider="openai",
    embedding_config={
        "model": "text-embedding-3-large",
        "dimensions": 2048
    }
)

# Privacy-focused agent with Ollama
private_agent = MemAgent(
    instruction="Privacy-focused assistant.",
    embedding_provider="ollama",
    embedding_config={
        "model": "nomic-embed-text"
    }
)
```

### Direct Embedding Manager Usage

```python
from memorizz.embeddings import EmbeddingManager

# Create embedding manager
manager = EmbeddingManager("openai", {
    "model": "text-embedding-3-small",
    "dimensions": 512
})

# Generate embeddings
embedding = manager.get_embedding("Hello world")
print(f"Dimensions: {manager.get_dimensions()}")
print(f"Provider info: {manager.get_provider_info()}")
```

### Backward Compatibility

Existing code continues to work without changes:

```python
# This still works - uses the globally configured provider
from memorizz.embeddings import get_embedding, get_embedding_dimensions

embedding = get_embedding("Some text")
dimensions = get_embedding_dimensions()
```

## Best Practices

### Choosing a Provider

**Use OpenAI when:**
- You need highest quality embeddings
- You're okay with API costs and internet dependency
- You want configurable dimensions
- You're building commercial applications

**Use Ollama when:**
- Privacy is a primary concern
- You want to avoid API costs
- You need offline operation
- You're building internal tools

**Use VoyageAI when:**
- You need very high quality embeddings
- You want specialized domain models (code, finance, law)
- You need multimodal capabilities (text + images)
- You require advanced features like contextualized embeddings
- You want configurable dimensions and data types

### Dimension Selection

**For OpenAI models:**
- **128-256**: Very efficient, good for basic similarity
- **512-1024**: Balanced quality and performance
- **1536+**: Maximum quality for critical applications

**For Ollama models:**
- Use default dimensions (fixed per model)
- Choose models based on your quality/speed requirements

### Performance Optimization

1. **Use smaller dimensions for development/testing**
2. **Configure dimensions once globally rather than per-agent**
3. **Cache embedding managers when possible**
4. **Monitor vector database performance with different dimensions**

## Migration Guide

### From Direct OpenAI Usage

**Before:**
```python
from memorizz.embeddings.openai import get_embedding
embedding = get_embedding(text, model="text-embedding-3-small")
```

**After:**
```python
from memorizz.embeddings import configure_embeddings, get_embedding

# Configure once
configure_embeddings("openai", {"model": "text-embedding-3-small"})

# Use as before
embedding = get_embedding(text)
```

### From Mixed Provider Usage

**Before:**
```python
from memorizz.embeddings.openai import get_embedding as openai_embed
from memorizz.embeddings.ollama import get_embedding as ollama_embed

# Use different functions
openai_result = openai_embed(text)
ollama_result = ollama_embed(text)
```

**After:**
```python
from memorizz.embeddings import EmbeddingManager

# Create managers for different providers
openai_manager = EmbeddingManager("openai")
ollama_manager = EmbeddingManager("ollama")

# Use consistent interface
openai_result = openai_manager.get_embedding(text)
ollama_result = ollama_manager.get_embedding(text)
```

## Error Handling

The system provides clear error messages for common issues:

```python
try:
    configure_embeddings("invalid_provider")
except ValueError as e:
    print(f"Unsupported provider: {e}")

try:
    configure_embeddings("openai", {"model": "invalid-model"})
except ValueError as e:
    print(f"Invalid model: {e}")

try:
    configure_embeddings("openai", {
        "model": "text-embedding-3-small",
        "dimensions": 5000  # Too large
    })
except ValueError as e:
    print(f"Invalid dimensions: {e}")
```

## Adding New Providers

To add a new embedding provider with the organized folder structure:

### Step 1: Create Provider Folder
```bash
mkdir src/memorizz/embeddings/your_provider
```

### Step 2: Create Package Files
```python
# src/memorizz/embeddings/your_provider/__init__.py
"""
Your Provider Embedding Provider

This package contains the Your Provider embedding implementation.
"""

from .provider import YourProviderEmbeddingProvider

__all__ = ['YourProviderEmbeddingProvider']
```

### Step 3: Implement Provider Class
```python
# src/memorizz/embeddings/your_provider/provider.py
import logging
from typing import List, Dict, Any
from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

class YourProviderEmbeddingProvider(BaseEmbeddingProvider):
    """Your Provider embedding provider implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Your initialization code
        self.model = self.config.get("model", "default-model")
        # ... other setup

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        # Your implementation
        return [0.1, 0.2, 0.3]  # Example

    def get_dimensions(self) -> int:
        return 768  # Your model's dimensions

    def get_default_model(self) -> str:
        return self.model
```

### Step 4: Register Provider
```python
# In src/memorizz/embeddings/__init__.py

# Add to EmbeddingProvider enum
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    YOUR_PROVIDER = "your_provider"  # Add this

# Add to EmbeddingManager._create_provider()
def _create_provider(self) -> BaseEmbeddingProvider:
    if self.provider_type == EmbeddingProvider.OPENAI:
        from .openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(self.config)
    elif self.provider_type == EmbeddingProvider.OLLAMA:
        from .ollama import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider(self.config)
    elif self.provider_type == EmbeddingProvider.YOUR_PROVIDER:  # Add this
        from .your_provider import YourProviderEmbeddingProvider
        return YourProviderEmbeddingProvider(self.config)
    # ... rest of implementation
```

### Step 5: Use Your Provider
```python
from memorizz.embeddings import configure_embeddings

# Configure your new provider
configure_embeddings("your_provider", {
    "model": "your-model-name",
    "api_key": "your-api-key"
})
```

This folder-based approach keeps each provider's code organized and makes the system easily extensible.

## Troubleshooting

### Common Issues

1. **"Ollama not available"**: Make sure Ollama is running (`ollama serve`)
2. **"Model not found"**: Pull the model first (`ollama pull nomic-embed-text`)
3. **"Invalid dimensions"**: Check model limits (3-small: max 1536, 3-large: max 3072)
4. **"API key not found"**: Set `OPENAI_API_KEY` environment variable
5. **"Connection refused"**: Check Ollama server URL and port

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("memorizz.embeddings").setLevel(logging.DEBUG)
```

This will show detailed information about embedding provider initialization and usage.

## Vector Index Dimension Synchronization

A critical feature of the embedding configuration system is **automatic vector index dimension synchronization**. This ensures that MongoDB vector indexes are created with the correct dimensions that match your configured embedding provider.

### How It Works

1. **Dynamic Dimension Detection**: When vector indexes are created, the system automatically queries the configured embedding provider for its dimensions
2. **Consistent Indexing**: All vector indexes (memory components, personas, toolbox, etc.) use the same dimensions
3. **No Manual Configuration**: You don't need to manually specify dimensions when creating indexes

### Before (Manual Dimension Management)

```python
# Old way - dimensions were hardcoded or manually specified
vector_index_definition = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1536,  # Hardcoded!
            "similarity": "cosine"
        }
    ]
}
```

### After (Automatic Dimension Synchronization)

```python
# New way - dimensions automatically match your embedding provider
from memorizz.embeddings import configure_embeddings
from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig

# Configure your embedding provider
configure_embeddings("openai", {
    "model": "text-embedding-3-small",
    "dimensions": 512
})

# Create memory provider - indexes automatically use 512 dimensions
memory_provider = MongoDBProvider(MongoDBConfig(
    uri="mongodb://localhost:27017"
))
```

### Benefits

1. **No Dimension Mismatches**: Vector operations will never fail due to dimension mismatches
2. **Automatic Updates**: Change your embedding configuration, and indexes will use the new dimensions
3. **Consistency**: All components use the same embedding dimensions
4. **Simplified Setup**: No need to manually track and specify dimensions

### Validation

You can validate that your vector indexes will use the correct dimensions:

```python
from memorizz.embeddings import configure_embeddings, get_embedding_dimensions

# Configure your provider
configure_embeddings("openai", {"model": "text-embedding-3-small", "dimensions": 256})

# Check dimensions
dimensions = get_embedding_dimensions()
print(f"Vector indexes will use {dimensions} dimensions")

# All subsequent vector index creation will use these dimensions automatically
```

### Important Notes

- **Existing Indexes**: If you change embedding dimensions, you may need to recreate existing vector indexes
- **Cross-Provider Consistency**: If using multiple embedding providers, ensure they have compatible dimensions
- **Performance Impact**: Larger dimensions provide better quality but require more storage and compute

This automatic synchronization ensures that your vector database operations work seamlessly regardless of which embedding provider or configuration you choose.
