# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""
VoyageAI Embedding Provider

This package contains the VoyageAI embedding provider implementation with support for:
- Text embeddings with multiple models and configurable dimensions
- Multimodal embeddings for text and images
- Contextualized chunk embeddings for documents
"""

from .provider import VoyageAIEmbeddingProvider

__all__ = ["VoyageAIEmbeddingProvider"]
