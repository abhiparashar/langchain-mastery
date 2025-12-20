from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import UsageMetadataCallbackHandler

def main():
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    store = {}

    def get_session_history(session_id:str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    chat = RunnableWithMessageHistory(model, get_session_history)

    tracker = UsageMetadataCallbackHandler()

    config = {"callbacks":[tracker], "configurable":{"session_id":"main"}}

    print("Chat with memory + token tracking")
    print("Commands: /usage, /clear, /quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break
        
        if user_input.lower == "clear":
            store["main"].clear()
            print("History cleared.\n")
            continue

        if user_input.lower == "usage":
            print("\n=== Token Usage ===")
            for model_name, usage in tracker.usage_metadata.items():
                print(f"{model_name}: {usage.get('total_tokens', 0):,} tokens")
            print()
            continue

        # Stream response
        print("Assistant: ", end="")
        for chunk in chat.stream(user_input, config=config):
            print(chunk.content, end="", flush=True)
        print("\n")


        


if __name__ == "__main__":
    main()