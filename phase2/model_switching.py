from langchain.chat_models import init_chat_model

class ChatSession:
    """Model switching"""
    def __init__(self, model_str):
        self.model = init_chat_model(model_str)
        self.model_str = model_str
    
    def chat(self, message:str) -> str:
        """Send message and stream response."""
        response=""
        for chunk in self.model.stream(message):
            print(chunk.content, end="", flush=True)
            response += chunk.content
            print()
        return response

    def switch_model(self, model_str) -> str:
        """Switch to a different model."""
        self.model = init_chat_model(model_str)
        self.model_string = model_str
        print(f"Switched to: {model_str}")


# Usage
session = ChatSession("google_genai:gemini-2.5-flash")
session.chat("Hello!")

session.switch_model("anthropic:claude-sonnet-4-20250514")
session.chat("Hello again!")