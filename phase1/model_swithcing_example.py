from langchain.chat_models import init_chat_model

MODELS = {
    "gemini": "google_genai:gemini-2.5-flash",
    "claude": "anthropic:claude-sonnet-4-20250514",
    "gpt": "openai:gpt-4o",
}

def main():
    """Start with default model"""
    current_model_key = "gemini"
    model = init_chat_model(MODELS[current_model_key])
    print(f"Current model is: {model}")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        # Simple model switch command
        if user_input.startswith("/model "):
            new_model = user_input.split(" ", 1)[1]
            if new_model in MODELS:
                model = init_chat_model(MODELS[new_model])
                current_model_key = new_model
                print(f"Switched to: {new_model}")
            else:
                print(f"Unknown model. Available: {list(MODELS.keys())}")
            continue

        # Stream response
        print("\nAssistant: ", end="")
        for chunk in model.stream(user_input):
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()