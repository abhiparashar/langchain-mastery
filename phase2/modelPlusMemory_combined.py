from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Create model
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

# 2. Set up memory store
store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. Wrap model with memory
chat = RunnableWithMessageHistory(model, get_session_history)

# 4. Use it (same session_id = remembers conversation)
config = {"configurable":{"session_id":"user-1"}}

response1 = chat.invoke("My name is Alice", config=config)
print(response1.content)

response2 = chat.invoke("What's my name?", config=config)
print(response2.content)
