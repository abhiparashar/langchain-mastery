from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import UsageMetadataCallbackHandler

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

store = {}

def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(model, get_session_history)

tracker = UsageMetadataCallbackHandler()

config = {"callbacks":[tracker], "configurable":{"session_id":"iuser_123"}}

chat.invoke("My name is Alice", config=config)
chat.invoke("What's my name?", config=config)

print(tracker.usage_metadata)
