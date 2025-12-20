from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

# 1. Create model
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# 2. Create callback to track usage
callbacks = UsageMetadataCallbackHandler()

# 3. Pass callback in config
config = {"callbacks":[callbacks]}

# 4. Use model (usage accumulates automatically)
model.invoke("Hello", config=config)
model.invoke("What is Python?", config=config)
model.invoke("Explain briefly", config=config)

# 5. Check accumulated usage
print(callbacks.usage_metadata)