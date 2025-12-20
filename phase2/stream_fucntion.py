from langchain.chat_models import init_chat_model

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

message = "What is Python?"

def stream_function(model, message: str) ->str :
    """Streaming with inti model"""
    response = ""
    for chunk in model.stream(message):
        response += chunk.content
    return response    

response = print(stream_function(model,message))
print(response, end="", flush=True)