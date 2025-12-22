from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from schema import SentimentResult
from prompts import prompt

def create_chain(
        model = "gemini-2.5-flash",
        model_provider = "google_genai",
        temperature = 0.1
):
    model = init_chat_model(model=model, model_provider=model_provider, temperature=temperature, max_retries=3, timeout=30)

    structured_model = model.with_structured_output(SentimentResult)

    chain = prompt | structured_model

    return chain