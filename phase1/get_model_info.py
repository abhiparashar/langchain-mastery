from langchain.chat_models import init_chat_model

def get_model_info(model) ->dict :
     """Extract model info for display."""
     return {
        "class": type(model).__name__,
        "model": getattr(model, "model", "unknown"),
        "temperature": getattr(model, "temperature", None),
        "supports_tools": getattr(model, "bind_tools"),
        "supports_streaming": getattr(model, "stream"),
     }


# Usage
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
print(get_model_info(model))