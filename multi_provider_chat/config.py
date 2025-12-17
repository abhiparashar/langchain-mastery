"""
Configuration for Multi-Provider Chat Application
LangChain 1.0+ compatible settings
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
     """Configuration for a specific model"""
     provider: str
     model_id:str
     display_name:str
     input_cost_per_1k:float # USD per 1K input tokens
     output_cost_per_1k:float # USD per 1K output tokens
     max_tokens:int = 4096
     temperature:int = 0.7

# Model configurations with 2025 pricing (approximate)
MODELS = {
    # OpenAI Models
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        max_tokens=4096,
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        max_tokens=16384,
    ),
    "gpt-4.1": ModelConfig(
        provider="openai",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.008,
        max_tokens=32768,
    ),
    "gpt-4.1-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        input_cost_per_1k=0.0004,
        output_cost_per_1k=0.0016,
        max_tokens=32768,
    ),
    
    # Anthropic Models
    "claude-sonnet-4": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=8192,
    ),
    "claude-sonnet-4.5": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=8192,
    ),
    "claude-haiku-4.5": ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        max_tokens=8192,
    ),
    
    # Ollama Models (local - no cost)
    "llama3.2": ModelConfig(
        provider="ollama",
        model_id="llama3.2",
        display_name="Llama 3.2 (Local)",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_tokens=4096,
    ),
    "mistral": ModelConfig(
        provider="ollama",
        model_id="mistral",
        display_name="Mistral (Local)",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_tokens=4096,
    ),
    "qwen2.5": ModelConfig(
        provider="ollama",
        model_id="qwen2.5",
        display_name="Qwen 2.5 (Local)",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        max_tokens=4096,
    ),

    #Gemini models
    "gemini-2.5-flash": ModelConfig(
    provider="google_genai",
    model_id="gemini-2.5-flash",
    display_name="Gemini 2.5 Flash",
    input_cost_per_1k=0.00015,
    output_cost_per_1k=0.0006,
    max_tokens=8192,
    ),
}

# Application settings
APP_SETTINGS = {
    "default_provider": "google_genai",
    "chat_history_file": "chat_history.json",
    "max_history_messages": 50,
    "stream_enabled": True,
}

def get_model_config(model_key: str) -> Optional[ModelConfig]:
    return MODELS.get(model_key)

def list_models_by_provider(provider:str) -> list[str]:
     return [key for key,config in MODELS.items() if config.provider==provider]

def get_available_providers() -> list[str]:
    return list(set(config.provider for config in MODELS.values()))

