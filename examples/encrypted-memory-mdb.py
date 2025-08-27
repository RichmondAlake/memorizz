from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider
import os

from bson import ObjectId
from pymongo import MongoClient
from pymongo.encryption import ClientEncryption, Algorithm
from bson.binary import Binary, STANDARD
from bson.codec_options import CodecOptions

local_master_key_string = os.urandom(96)
os.environ["VOYAGE_API_KEY"] = ""

# --- Step 2: Configure KMS and Key Vault ---
# The master key is used to encrypt the data keys. For this local demo,
# we use a randomly generated 96-byte key. In a production environment,
# this key would be managed by a secure Key Management Service (KMS)
# like AWS KMS, Azure Key Vault, or GCP KMS. 
# A separate MongoClient is used for the key vault for security best practices.
# Configure the local KMS provider
kms_providers = {"local": {"key": local_master_key_string}}

# Define the namespace for the key vault collection, which will store the data keys.
key_vault_namespace = "encryption.__pymongoTestKeyVault"

# Initialize a separate client for the key vault. This is a security best practice.
key_vault_client = MongoClient("mongodb://localhost:27017/?directConnection=true")

# The MongoDBConfig object encapsulates all necessary settings: URI, database name,
# embedding provider details for vector search, and the critical encryption configuration.
# The `encryption_config` dictionary tells our provider:
#   - Where to find the master key (kms_providers)
#   - Where to store the data keys (key_vault_namespace)
#   - Which fields to encrypt for specific collections (`field_mappings`)
# The algorithms are specified per field: `Random` encryption is used here because
# we are encrypting a complex BSON object (`function` and `type`), which is
# not queryable and therefore does not require `Deterministic` encryption.
mongodb_config = MongoDBConfig(
    uri="mongodb://localhost:27017/?retryWrites=true&w=majority&directConnection=true",
    db_name="testing_memorizz",
    embedding_provider="voyageai",
    embedding_config={
        "embedding_type": "text",
        "model": "voyage-3.5",
        "output_dimension": 1024,
        "input_type": "query"
    },
    encryption_config={
        "kms_providers": kms_providers,
        "key_vault_namespace": key_vault_namespace,
        "key_vault_client": key_vault_client,
        "field_mappings": {
            "toolbox": {
                "encrypted_fields": {
                    "function": Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random,
                    "type": Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random
                }
            },
            "personas": {
                "encrypted_fields": {
                    "background": Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random
                }
            }
        }
    }
)

# Create a memory provider
memory_provider = MongoDBProvider(mongodb_config)

from functools import lru_cache
from yahooquery import Ticker
import time

@lru_cache(maxsize=128)
def _fetch_price(symbol: str) -> float:
    """
    Internal helper to fetch the latest market price via yahooquery.
    Caching helps avoid repeated hits for the same symbol.
    """
    ticker = Ticker(symbol)
    # This returns a dict keyed by symbol:
    info = ticker.price or {}
    # regularMarketPrice holds the current trading price
    price = info.get(symbol.upper(), {}).get("regularMarketPrice")
    if price is None:
        raise ValueError(f"No price data for '{symbol}'")
    return price

def get_stock_price(
    symbol: str,
    currency: str = "USD",
    retry: int = 3,
    backoff: float = 0.5
) -> str:
    """
    Get the current stock price for a given symbol using yahooquery,
    with simple retry/backoff to handle occasional rate-limits.

    Parameters
    ----------
    symbol : str
        Stock ticker, e.g. "AAPL"
    currency : str, optional
        Currency code (Currently informational only; yahooquery returns native)
    retry : int, optional
        Number of retries on failure (default: 3)
    backoff : float, optional
        Backoff factor in seconds between retries (default: 0.5s)

    Returns
    -------
    str
        e.g. "The current price of AAPL is 172.34 USD."
    """
    symbol = symbol.upper()
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            price = _fetch_price(symbol)
            return f"The current price of {symbol} is {price:.2f} {currency.upper()}."
        except Exception as e:
            last_err = e
            # simple backoff
            time.sleep(backoff * attempt)
    # if we get here, all retries failed
    raise RuntimeError(f"Failed to fetch price for '{symbol}' after {retry} attempts: {last_err}")

import requests

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

from memorizz.long_term_memory.procedural.toolbox import Toolbox

from memorizz.embeddings import configure_embeddings, get_embedding
configure_embeddings(
    provider="voyageai",
    config={
        "embedding_type": "text",
        "model": "voyage-3.5",
        "output_dimension": 1024,
        "input_type": "query"
    }
)

os.environ["OPENAI_API_KEY"] = ""

toolbox = Toolbox(
    memory_provider=memory_provider,
)
# Now the tools are registered in the memory provider within the toolbox
toolbox.register_tool(get_weather)
toolbox.register_tool(get_stock_price)
print(toolbox.list_tools())