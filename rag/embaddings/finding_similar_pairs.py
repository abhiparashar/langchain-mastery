from langchain.embeddings import init_embeddings
import numpy as np

embeddings = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")

def build_similarity_matrix(vectors:list)->np.ndarray:
    """
    Build a matrix showing similarity between all pairs.
    matrix[i][j] = similarity between text i and text j
    """
    emb_array = np.array(vectors)

    # Normalize each embedding
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    normalized = emb_array/norms

    # Similarity matrix = normalized @ normalized.T
    return np.dot(normalized, normalized.T)

def find_most_similar_pairs(texts: list[str], matrix: np.ndarray, top_k: int = 3):
    """Find the top-k most similar pairs (excluding self-similarity)."""
    n = len(texts)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "text1": texts[i],
                "text2": texts[j],
                "similarity": matrix[i][j]
            })
    
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_k]

# Example usage
texts = [
    "The weather is beautiful today",
    "It's a sunny and pleasant day",
    "Machine learning is fascinating",
    "AI and deep learning are amazing",
    "I need to buy groceries",
    "Shopping for food at the store"
]

print("Generating embeddings...")
vectors = embeddings.embed_documents(texts)

print("Building similarity matrix...")
matrix = build_similarity_matrix(vectors)

print("\n Similarity Matrix (first 3x3):")
print(np.round(matrix[:3, :3], 3))


print("\n🏆 Top 3 Most Similar Pairs:")
top_pairs = find_most_similar_pairs(texts, matrix, top_k=3)

for i, pair in enumerate(top_pairs, 1):
    print(f"\n{i}. Similarity: {pair['similarity']:.4f}")
    print(f"   Text A: '{pair['text1']}'")
    print(f"   Text B: '{pair['text2']}'")