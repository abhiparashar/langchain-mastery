from langchain.embeddings import init_embeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")

text = "I love machine learning"

embedding = embeddings.embed_query(text)

print(f"Text: '{text}'")
print(f"Embedding length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
print(f"Type: {type(embedding)}")