# Multi-Provider Chat Application

A CLI chat application built with **LangChain 1.0+** patterns, supporting multiple LLM providers with a unified interface.

## Features

- 🔄 **Multi-Provider Support**: OpenAI, Anthropic, and Ollama via `init_chat_model`
- 📋 **Model Profiles**: Check capabilities using `.profile` attribute
- 🌊 **Streaming**: Real-time responses with `.stream()` and `.astream()`
- 🧠 **Conversation Memory**: Session-based history with `RunnableWithMessageHistory`
- 📊 **Token Tracking**: Usage counting with `UsageMetadataCallbackHandler`
- 💰 **Cost Estimation**: Real-time cost tracking per request
- 📤 **Export**: Save chat history to JSON

## LangChain 1.0 Patterns Used

### 1. `init_chat_model` - Unified Model Initialization

```python
from langchain.chat_models import init_chat_model

# Provider:model format
model = init_chat_model("openai:gpt-4o")
model = init_chat_model("anthropic:claude-sonnet-4-20250514")
model = init_chat_model("ollama:llama3.2")

# Configurable model (switch at runtime)
configurable = init_chat_model(
    "openai:gpt-4o",
    configurable_fields=("model", "model_provider", "temperature")
)
response = configurable.invoke(
    "Hello",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}}
)
```

### 2. Model Profiles - Check Capabilities

```python
model = init_chat_model("openai:gpt-4o")

# Check profile for capabilities
if hasattr(model, 'profile') and model.profile:
    if model.profile.structured_output:
        # Use native structured output
        pass
    if model.profile.tool_calling:
        # Bind tools
        pass
```

### 3. `.content_blocks` - Standard Content Format

```python
response = model.invoke("What's the weather?")

# Access standardized content blocks
for block in response.content_blocks:
    if block["type"] == "text":
        print(block["text"])
    elif block["type"] == "tool_call":
        print(f"Tool: {block['name']}({block['args']})")
```

### 4. `RunnableWithMessageHistory` - Conversation Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Session store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Use with session
response = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "alice_session"}}
)
```

### 5. `UsageMetadataCallbackHandler` - Token Tracking

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()
response = model.invoke("Hello", config={"callbacks": [callback]})

# Get usage per model
for model_name, usage in callback.usage_metadata.items():
    print(f"Input tokens: {usage['input_tokens']}")
    print(f"Output tokens: {usage['output_tokens']}")
```

## Installation

```bash
# Clone/create the project
cd multi_provider_chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set environment variables for your providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Ollama (no key needed, just run ollama locally)
# ollama serve
# ollama pull llama3.2
```

## Usage

### Basic Usage

```bash
# Start with default model (gpt-4o-mini)
python main.py

# Use specific model
python main.py --model gpt-4o
python main.py --model claude-sonnet-4
python main.py --model llama3.2

# Use provider's default model
python main.py --provider anthropic

# Disable streaming
python main.py --no-stream

# Use async streaming
python main.py --async

# List available models
python main.py --list-models
```

### Interactive Commands

Once in the chat:

```
/help           - Show available commands
/models         - List all available models
/switch [model] - Switch to a different model
/history        - View conversation history
/clear          - Clear conversation history
/usage          - Show token usage and cost
/export [file]  - Export chat to JSON
/profile        - Show model capabilities
/quit           - Exit application
```

### Example Session

```
============================================================
Model: GPT-4o Mini
Provider: openai
Session: session_20241216_143022
============================================================
Capabilities: tools, structured output, vision
Streaming: enabled
============================================================

Type your message or /help for commands. /quit to exit.

You: Hello! My name is Alice.
```
