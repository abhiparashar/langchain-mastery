from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

with get_usage_metadata_callback() as cb:
    model.invoke("Hello")
    model.invoke("Tell me a joke")

print(cb.usage_metadata)