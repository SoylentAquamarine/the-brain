"""
adapters/__init__.py — Auto-discovery registry for all provider adapters.

Import the registry dict to get a map of { provider_key: adapter_instance }
for all adapters that have valid credentials available at runtime.

Usage:
    from brain.adapters import REGISTRY
    groq = REGISTRY.get("groq")
"""

from brain.adapters.anthropic_adapter import AnthropicAdapter
from brain.adapters.openai_adapter     import OpenAIAdapter
from brain.adapters.gemini_adapter     import GeminiAdapter
from brain.adapters.groq_adapter       import GroqAdapter
from brain.adapters.cohere_adapter     import CohereAdapter

# All known adapters — add new ones here as the project grows.
ALL_ADAPTERS = [
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    GroqAdapter,
    CohereAdapter,
]

# Instantiate each and keep only those that report is_available() == True.
REGISTRY: dict = {
    cls.PROVIDER_KEY: cls()
    for cls in ALL_ADAPTERS
}

__all__ = ["REGISTRY", "ALL_ADAPTERS"]
