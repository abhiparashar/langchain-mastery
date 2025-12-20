from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import MODELS, DEFAULT_MODEL
from model_factory import create_model

_store = {}

def get_session_history(session_id:str):
    """Get or create chat history for a session."""
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory
    return _store[session_id]

def wrap_with_memory(model_key:str):
    """Wrap a model with message history."""
    return RunnableWithMessageHistory(create_model(model_key), get_session_history)

def clear_session(session_id:str):
    """Clear history for a session."""
    if session_id in _store:
        _store[session_id].clear()

def list_sessions() -> list[str]:
    """List all active session IDs."""
    list(_store.keys())

def get_message_count(session_id: str) ->int:
    """Get number of messages in a session."""
    if session_id not in _store:
        return 0
    return len(_store[session_id].messages)

def export_session(session_id: str) -> list[dict]:
    """Export session history as list of dicts."""
    if session_id not in _store:
        return []
    
    messages = []

    for msg in _store[session_id].messages:
        messages.append({
            "role": msg.type,  # "human" or "ai"
            "content": msg.content,
        })
        
    return messages