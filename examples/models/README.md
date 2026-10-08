# New embedding and decision models

EmbeddingGemma 2 is available through the Hugging Face embedding provider:

```python
from memorizz.embeddings.huggingface.provider import HuggingFaceEmbeddingProvider

embeddings = HuggingFaceEmbeddingProvider({"model": "google/embeddinggemma-2"})
document = embeddings.get_embedding("Mira owns Harbor.")
query = embeddings.get_embedding("Who owns Harbor?", input_type="query")
```

Install `memorizz[huggingface]`. Text stores load only the text encoder, use
normalized 768-dimensional vectors, and select Document/SearchQuery prompts.
Set `dimensions` to 512, 256 or 128 for a consistently truncated index. Rebuild
an existing index when changing its embedding model or dimensions.

OpenAI Decisions uses its dedicated API rather than chat completions:

```python
from memorizz import OpenAIDecisions

result = OpenAIDecisions().evaluate(
    "The launch owner changed from Mina to Mira.",
    [{"name": "changed", "type": "predicate", "instructions": "Did the owner change?"}],
)
```

Configure `OPENAI_API_KEY` in your environment. Predicate, typed choice and
ordered score questions are supported, with refusal and probability validation.
Evalground's reranker selector includes OpenAI Decisions alongside Jev; both
support predicate (Jev `noul`), choice and score recipes. Cost accounting uses
the Decisions endpoint's pricing, separately from chat-model pricing.

Run the paired synthetic relevance example with `OPENAI_API_KEY` and
`TYPESAFE_API_KEY` configured, or enter the Jev key at the hidden terminal prompt:

```sh
python examples/models/decisions_vs_jev.py --output /tmp/decisions-vs-jev.json
```

It records top-1 relevance, latency, reported token usage and estimated API cost.
The three cases verify integrations; they do not establish production accuracy
or an overall winner. Credential values are never saved in its output.

Sources: [EmbeddingGemma 2 guide](https://developers.googleblog.com/en/embeddinggemma-2-the-developer-guide/),
[OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions),
[Jev API](https://docs.typesafe.ai/api).
