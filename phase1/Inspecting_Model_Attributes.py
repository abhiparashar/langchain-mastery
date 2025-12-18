from langchain.chat_models import init_chat_model

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# Basic attributes (vary by provider)
print(model.model)           # Model name
print(model.temperature)     # Temperature setting
print(type(model).__name__)  # Class type: ChatGoogleGenerativeAI