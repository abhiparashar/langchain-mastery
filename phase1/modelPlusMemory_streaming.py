from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable":{"session_id":"user-1"}}

# Streaming works the same way!
for chunk in chat.stream(config=config):
    print(chunk.content, end="", flush=True)
    print()

for chunk in chat.stream("What's my name?", config=config):
    print(chunk.content, end="", flush=True)
print()
