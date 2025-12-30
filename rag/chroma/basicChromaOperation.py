import chromadb
from langchain_chroma import Chroma
from langchain.embeddings import init_embeddings

embeddings_model = init_embeddings("huggingface:sentence-transformers/all-MiniLM-L6-v2")


# STEP 2: Create Chroma Database (In-Memory)
vector_store = Chroma(collection_name="my_recipes",
                      embedding_function=embeddings_model
                    )

# STEP 3: Add Documents
documents = [
    "How to make masala chai with ginger and cardamom",
    "Cold coffee recipe with ice cream",
    "Green tea health benefits and brewing tips",
    "Mango lassi summer drink recipe",
    "Hot chocolate for winter evenings"
]

# Add documents with auto-generated IDs
vector_store.add_texts(texts=documents)
print("✅ Documents added successfully!")

# Method 2: With Metadata (Recommended ⭐)
documents_with_meta = [
    "Samosa recipe with potato filling",
    "Chocolate cake baking guide",
    "Paneer tikka grilling method"
]

metadatas = [
    {"category": "snack", "cuisine": "indian", "difficulty": "medium"},
    {"category": "dessert", "cuisine": "western", "difficulty": "hard"},
    {"category": "starter", "cuisine": "indian", "difficulty": "easy"}
]

ids = ["recipe_1", "recipe_2", "recipe_3"]

vector_store.add_texts(texts=documents_with_meta,
                       metadatas=metadatas,
                       ids=ids
                    )
print("✅ Documents with metadata added!")

# SIMILARITY SEARCH - Basic
# Simple search - returns Document objects
query = "I want something hot to drink"
results = vector_store.similarity_search(query, k=3)

print("🔍 Search Results:")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")


# SIMILARITY SEARCH - With Scores
# When you need to know HOW similar the results are
results_with_scores  = vector_store.similarity_search_with_score(query, k=3)

print("\n🔍 Results with Similarity Scores:")
for doc, score in results_with_scores:
    print(f"\n📄 {doc.page_content}")
    print(f"   Score: {score:.4f}")  # Lower score = More similar (distance)
    print(f"   Metadata: {doc.metadata}")
