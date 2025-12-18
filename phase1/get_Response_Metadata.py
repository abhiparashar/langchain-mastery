from langchain.chat_models import init_chat_model

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

response = model.invoke("Hi")

# Response metadata contains useful info
print(response.response_metadata)
# May include: model_name, finish_reason, token usage, etc.

print(response.usage_metadata)
# {'input_tokens': 2, 'output_tokens': 5, 'total_tokens': 7}


