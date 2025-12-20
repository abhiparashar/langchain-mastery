from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class MemoryChat:
    def __init__(self, model_str:str, session_id:str = "default"):
        self.model = init_chat_model(model_str)
        self.session_id = session_id
        self._store = {}
        self.chat = RunnableWithMessageHistory(
            self.model,
            self._get_session_history
        )

    def _get_session_history(self, session_id:str):
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]
        
    @property
    def _config(self):
        return {"configurable": {"session_id": self.session_id}}

    def send(self, message:str) -> str:
        """Send message, stream response, return full text."""
        response = ""
        for chunk in self.chat.stream(message, config=self._config):
            print(chunk.content, end="", flush=True)
        response += chunk.content
        return response

    def clear_history(self):
        """Clear conversation history."""
        if self.session_id in self._store:
            self._store[self.session_id].clear()
            print("History cleared.")



# Usage
chat = MemoryChat("google_genai:gemini-2.5-flash")
chat.send("I'm learning Python")
chat.send("What am I learning?")  # "You're learning Python!"
chat.clear_history()