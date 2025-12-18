import asyncio
from langchain.chat_models import init_chat_model

async def stream_async():
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
    async for chunk in model.astream("Hello"):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_async())       
