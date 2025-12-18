from langchain.chat_models import init_chat_model; 

model = init_chat_model("google_genai:gemini-2.5-flash")

response = model.invoke("say hello in 3 languages")

print(response.content)