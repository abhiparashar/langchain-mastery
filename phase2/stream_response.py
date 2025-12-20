from langchain.chat_models import init_chat_model

def stream_response(model, prompt:str) -> str:
    """Stream response to terminal, return full text."""
    full_text = ""
    for chunk in model.stream(prompt):
        print(chunk.content, end="", flush=True)
        full_text += chunk.content
        print()
    return full_text


# Usage
model = init_chat_model("google_genai:gemini-2.5-flash")
response = stream_response(model, "Explain streaming in one sentence")
