from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def main():
    # Setup
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    store = {}

    def get_session_history(session_id:str) -> str:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    chat = RunnableWithMessageHistory(model, get_session_history)

    config = {"configurable": {"session_id": "main"}}

    print("Chat with memory (type /clear to reset, /quit to exit)\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "quit":
            break

        if user_input.lower() == "clear":
            store["main"].clear()
            print("History cleared.\n")
            continue

        # Stream response with memory
        print("Assistant: ", end="")
        for chunk in chat.stream(user_input, config):
            print(chunk.content, end="", flush=True)
            print("\n")

if __name__ =="__main__":
    main()