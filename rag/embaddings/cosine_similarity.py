from langchain.embeddings import init_embeddings
import numpy as np

# Initialize embeddings (same interface for any provider!)
embeddings = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")

def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Where:
    - A · B is the dot product
    - ||A|| is the magnitude (length) of vector A
    """
    a = np.array(vec1) 
    b = np.array(vec2)

    dot_product =  np.dot(a,b)
    magnitude_a  = np.linalg.norm(a)
    magnitude_b =  np.linalg.norm(b)

    return float(dot_product/(magnitude_a * magnitude_b))
    
# Test with example texts
text1 = "I love eating pizza"
text2 = "Pizza is my favorite food"
text3 = "The stock market crashed today"

# Get embeddings using LangChain
emb1 = embeddings.embed_query(text1)
emb2 = embeddings.embed_query(text2)
emb3 = embeddings.embed_query(text3)

print("Similarity Scores:")
print(f"'{text1}' vs '{text2}'")
print(f"  → {cosine_similarity(emb1, emb2):.4f}  (Should be HIGH - similar topic)")
print()
print(f"'{text1}' vs '{text3}'")
print(f"  → {cosine_similarity(emb1, emb3):.4f}  (Should be LOW - different topic)")
