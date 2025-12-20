from langchain.chat_models import init_chat_model
from config import MODELS, DEFAULT_MODEL

def create_model(model_key:str) -> str:
    """Create a chat model from a model key."""
    if model_key not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return init_chat_model(MODELS[model_key])

def get_default_model():
    """Create the default model."""
    return create_model(DEFAULT_MODEL)

def list_models():
    """List available model keys."""
    list(MODELS.keys())

def get_model_info(model_key:str):
     """Get info about a model."""
     if model_key not in MODELS:
         raise ValueError(f"Unknown model: {model_key}")
     
     return {
        "key":model_key,
        "model_string": MODELS[model_key]
    }

        
    