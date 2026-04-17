"""
adapters/__init__.py — Auto-discovery registry for all provider adapters.

Import REGISTRY to get a map of { provider_key: adapter_instance }
for all adapters that have valid credentials at runtime.
"""

from brain.adapters.anthropic_adapter import AnthropicAdapter
from brain.adapters.openai_adapter     import OpenAIAdapter
from brain.adapters.gemini_adapter     import GeminiAdapter
from brain.adapters.groq_adapter       import GroqAdapter
from brain.adapters.cohere_adapter     import CohereAdapter
from brain.adapters.mistral_adapter    import MistralAdapter
from brain.adapters.deepseek_adapter   import DeepSeekAdapter
from brain.adapters.cerebras_adapter   import CerebrasAdapter
from brain.adapters.together_adapter   import TogetherAdapter

ALL_ADAPTERS = [
    AnthropicAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    GroqAdapter,
    CohereAdapter,
    MistralAdapter,
    DeepSeekAdapter,
    CerebrasAdapter,
    TogetherAdapter,
]

REGISTRY: dict = {
    cls.PROVIDER_KEY: cls()
    for cls in ALL_ADAPTERS
}

__all__ = ["REGISTRY", "ALL_ADAPTERS"]
