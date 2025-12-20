from langchain.chat_models import init_chat_model

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# .stream() returns an iterator of chunks
for chunk in model.stream("Write a haiku about Python"):
    print(chunk.content, end="", flush=True)