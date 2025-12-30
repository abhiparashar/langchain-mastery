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

# ============================================
# UPDATE - Using underlying Chroma client
# ============================================

# LangChain's Chroma wrapper doesn't have direct update
# We need to access the underlying collection

# Get the underlying Chroma collection
collection = vector_store._collection

# Update document text and metadata
collection.update(
    ids=["recipe_1"],
    documents=["Easy samosa recipe - crispy and delicious"],
    metadatas=[{"category": "snack", "cuisine": "indian", "difficulty": "easy"}]
)
print("✅ Document updated!")

# Verify the update
results = vector_store.similarity_search("easy samosa", k=1)
print(f"Updated document: {results[0].page_content}")
print(f"Updated metadata: {results[0].metadata}")

# DELETE - Remove specific documents
# Method 1: Delete by IDs
vector_store.delete(ids=["recipe_2"])
print("✅ Deleted recipe_2")

# Method 2: Delete by filter (using underlying collection)
collection = vector_store._collection
collection.delete(
    where={"category": "dessert"}  # Delete all desserts
)
print("✅ Deleted all dessert recipes")

# Method 3: Delete all documents in collection
# WARNING: This deletes everything!
# vector_store.delete_collection()

# FILTERED QUERIES - Various Examples
# First, let's add more data with rich metadata
documents = [
    "Butter chicken curry recipe",
    "Pasta carbonara Italian style",
    "Sushi making guide Japanese cuisine",
    "Tacos Mexican street food",
    "Biryani Hyderabadi style",
    "Pizza Margherita authentic recipe",
    "Dosa South Indian breakfast",
    "Pad Thai noodles recipe"
]

metadatas = [
    {"cuisine": "indian", "type": "main", "spice_level": 3, "time_mins": 45},
    {"cuisine": "italian", "type": "main", "spice_level": 1, "time_mins": 30},
    {"cuisine": "japanese", "type": "main", "spice_level": 1, "time_mins": 60},
    {"cuisine": "mexican", "type": "snack", "spice_level": 2, "time_mins": 20},
    {"cuisine": "indian", "type": "main", "spice_level": 3, "time_mins": 90},
    {"cuisine": "italian", "type": "main", "spice_level": 1, "time_mins": 25},
    {"cuisine": "indian", "type": "breakfast", "spice_level": 2, "time_mins": 30},
    {"cuisine": "thai", "type": "main", "spice_level": 2, "time_mins": 25}
]

# Add to vector store
vector_store.add_texts(texts=documents, metadatas=metadatas)

# FILTER 1: Simple equality filter
# Find recipes, but only Indian cuisine
results = vector_store.similarity_search(
    query="delicious food recipe",
    k=5,
    filter={"cuisine": "indian"}  # Simple filter
)

print("🇮🇳 Indian Recipes:")
for doc in results:
    print(f"  • {doc.page_content}")

# FILTER 2: Using $in operator
# Find Asian cuisines only

results = vector_store.similarity_search(
    query="noodles and rice dishes",
    k=5,
    filter={"cuisine": {"$in": ["indian", "japanese", "thai"]}}
)

print("\n🌏 Asian Recipes:")
for doc in results:
    print(f"  • {doc.page_content} ({doc.metadata['cuisine']})")

# FILTER 3: Numeric comparison
# Find quick recipes (less than 30 minutes)
results = vector_store.similarity_search(
    query="quick easy recipe",
    k=5,
    filter={"time_mins": {"$lte": 30}}
)

print("\n⚡ Quick Recipes (≤30 mins):")
for doc in results:
    print(f"  • {doc.page_content} ({doc.metadata['time_mins']} mins)")

# FILTER 4: Combining with $and
# Find Indian recipes that are NOT too spicy
results = vector_store.similarity_search(
    query="tasty food",
    k=5,
    filter={
        "$and": [
            {"cuisine": "indian"},
            {"spice_level": {"$lte": 2}}
        ]
    }
)

print("\n🇮🇳🌶️ Mild Indian Recipes:")
for doc in results:
    print(f"  • {doc.page_content} (Spice: {doc.metadata['spice_level']})")

# FILTER 5: Using $or
# ============================================
# Find either Italian OR Mexican recipes

results = vector_store.similarity_search(
    query="cheesy delicious food",
    k=5,
    filter={
        "$or": [
            {"cuisine": "italian"},
            {"cuisine": "mexican"}
        ]
    }
)

print("\n🇮🇹🇲🇽 Italian or Mexican:")
for doc in results:
    print(f"  • {doc.page_content}")

# ============================================
# FILTER 6: Complex filter with $and and $or
# ============================================
# Find (Indian OR Italian) AND quick (≤30 mins)

results = vector_store.similarity_search(
    query="dinner recipe",
    k=5,
    filter={
        "$and": [
            {"$or": [{"cuisine": "indian"}, {"cuisine": "italian"}]},
            {"time_mins": {"$lte": 30}}
        ]
    }
)

print("\n🍝 Quick Indian/Italian Recipes:")
for doc in results:
    print(f"  • {doc.page_content} ({doc.metadata['time_mins']} mins)")

# ============================================
# FILTER 7: Not equal filter
# ============================================
# Find everything EXCEPT Indian cuisine

results = vector_store.similarity_search(
    query="food recipe",
    k=5,
    filter={"cuisine": {"$ne": "indian"}}
)

print("\n🌍 Non-Indian Recipes:")
for doc in results:
    print(f"  • {doc.page_content} ({doc.metadata['cuisine']})")
