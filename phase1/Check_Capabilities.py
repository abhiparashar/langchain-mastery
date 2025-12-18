from langchain.chat_models import init_chat_model;

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# Check if model supports tool calling
supports_tools  = hasattr(model, "bind_tools")

print(f"Supports tools: {supports_tools}")

# Check if model supports structured output
supports_structured  = hasattr(model, "with_structured_output")
print(f"Supports structured output: {supports_structured}")  # True
