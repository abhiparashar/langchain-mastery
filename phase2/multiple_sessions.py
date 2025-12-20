from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = init_chat_model("google_genai:gemini-2.5-flash")
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(model, get_session_history)

# User Alice
alice_config = {"configurable": {"session_id": "alice"}}
chat.invoke("My name is Alice", config=alice_config)

# User Bob (separate conversation)
bob_config = {"configurable": {"session_id": "bob"}}
chat.invoke("My name is Bob", config=bob_config)

# Each user has their own history
print(chat.invoke("What's my name?", config=alice_config))
print(chat.invoke("What's my name?", config=bob_config))
