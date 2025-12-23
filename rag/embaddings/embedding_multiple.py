from langchain.embeddings import init_embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embedding = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")

# Multiple texts → embed_documents
texts = [
    "The cat sat on the mat",
    "A kitten is sitting on a rug",
    "Python is a programming language",
    "JavaScript is used for web development"
]

# Get embeddings for all texts at once
vectors = embedding.embed_documents(texts)

for i, text in enumerate(vectors):
    print(f"Text {i+1}: '{text[:15]}...' → {len(vectors[i])} dimensions")