"""Multi-Provider Chat CLI."""

from config import MODELS, DEFAULT_MODEL, DEFAULT_SESSION_ID
from model_factory import create_model
from memory import wrap_with_memory, clear_session
from langchain_core.callbacks import UsageMetadataCallbackHandler

def main():
    # Initialize
    current_model_key = DEFAULT_MODEL
    chat = wrap_with_memory(current_model_key)
    tracker = UsageMetadataCallbackHandler()

    # Config for memory + tracking
    config = {
        "callbacks" :[tracker],
        "configurable" : {"session_id": DEFAULT_SESSION_ID}
    }

    print(f"Chat CLI - Using: {current_model_key}")
    print("Commands: /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            # Stream response
            print("Assistant: ", end="", flush=True)
            for chunk in chat.stream(user_input, config=config):
                print(chunk.content, end="", flush= True)
            print("\n")

        except KeyboardInterrupt:
            print("\n Goodbye !!")
        break

if __name__  == "__main__":
    main()