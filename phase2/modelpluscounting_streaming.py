from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

callbacks = UsageMetadataCallbackHandler()

config = {"callbacks":[callbacks]}

# Streaming works too!
for chunks in model.stream("Tell me a joke", config=config):
    print(chunks.content, end="", flush=True)
print()

print(f"\nTokens used: {callbacks.usage_metadata}")
