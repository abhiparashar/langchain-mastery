"""
Model Factory using LangChain 1.0 init_chat_model
Supports OpenAI, Anthropic, and Ollama with unified initialization
"""

from dataclasses import dataclass;
from typing import Optional, Any

from config import ModelConfig, MODELS, get_model_config
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

@dataclass
class ModelCapabilities:
     """Model capabilities extracted from profile"""
     supports_streaming : bool = True
     supports_tool_calling: bool = False
     supports_structured_output : bool  = False
     supports_image_input : bool = False
     supports_audio_input : bool = False
     max_input_tokens : Optional[int] = None
     max_output_tokens : Optional[int] = None

class ModelFactory:
    """
    Factory for creating chat models using LangChain 1.0 init_chat_model
    
    Usage:
        factory = ModelFactory()
        model = factory.create_model("gpt-4o")
        capabilities = factory.get_capabilities(model)
    """
    def __init__(self):
         self.model_cache = dict[str, BaseChatModel] = {}

    def create_model(
        self,
        model_key: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: bool = True,
        **kwargs
    ) -> BaseChatModel :
        """
        Create a chat model using init_chat_model (LangChain 1.0 pattern)
        
        Args:
            model_key: Key from MODELS config (e.g., "gpt-4o", "claude-sonnet-4")
            temperature: Override default temperature
            max_tokens: Override default max tokens
            streaming: Enable streaming support
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized BaseChatModel instance
        """

        config = get_model_config(model_key)
        if not config :
             raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        # Build the model identifier string (provider:model_id format)
        model_identifier = f"{config.provider}:{config.model_id}"

        # Prepare model parameters
        model_params = {
             "temperature" : temperature if temperature is not None else config.temperature,
             "max_tokens" : max_tokens if max_tokens is not None else config.max_tokens,
              **kwargs
        }

        # Add streaming parameter if supported
        if streaming:
             model_params[streaming] = True

         # Use init_chat_model for unified initialization
        model = init_chat_model(model_identifier,
                                  **model_params)
        return model
         