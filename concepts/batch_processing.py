"""
STEP 5: Batch Processing
========================

CONCEPT: Process multiple inputs efficiently with .batch() and async.

WHY THIS MATTERS:
- Sequential: 10 requests × 1 second = 10 seconds
- Parallel:   10 requests with batch = ~2 seconds (5x faster!)

KEY INSIGHT: Use .batch() for multiple inputs, not a for loop.

NOTE: Requires GOOGLE_API_KEY to run examples.
"""

import os
import time
import asyncio
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

# SETUP: Our Sentiment Schema and Chain
class Sentitype(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentResult(BaseModel):
    sentiment:Sentitype = Field(
        description="The sentiment"
    )
    confidence : float = Field(
        le=0.0,
        ge=1.0,
        description="Confidence 0-1"
    )

def create_chain():
    """Create our sentiment analysis chain."""
    prompt = ChatMessagePromptTemplate.format_messages([
        ("system","You are a sentiment analyzer. Be brief."),
        ("human", "Sentiment of: {text}")
    ])

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.3)

    structured_model = model.with_structured_output(SentimentResult)

    return prompt | structured_model

# PART A: Sequential vs Batch (Speed Comparison)
def demo_sequential_vs_batch():
    """Compare sequential processing vs batch processing."""
    print("\n⏱️ PART A: Sequential vs Batch Speed")
    print("-" * 40)

    chain = create_chain()

    texts = [
        "I love this!",
        "This is terrible.",
        "It's okay.",
        "Amazing product!",
        "Worst experience ever.",
    ]

    # Method 1: Sequential (SLOW)
    print("\n1. Sequential (for loop):")
    start = time.time()
    sequential_results = []
    for text in texts:
        result = chain.invoke({"text":text})
        sequential_results.append(result)
        sequential_time  = time.time() - start
        print(f"   Time: {sequential_time:.2f} seconds")
    
    # Method 2: Batch (FAST)
    print("\n2. Batch (parallel):")
    start = time.time()
    inputs = [{"text": t} for t in texts]
    batch_results  = chain.batch(inputs)
    batch_time = time.time() - start
    print(f"   Time: {batch_time:.2f} seconds")

    # Speedup
    speedup = sequential_time / batch_time if batch_time > 0 else 0
    print(f"\n   🚀 Speedup: {speedup:.1f}x faster with batch!")
    
    # Verify same results
    print("\n   Results match:", all(
        s.sentiment == b.sentiment 
        for s, b in zip(sequential_results, batch_results)
    ))

# PART B: Batch with Configuration
def demo_batch_config():
    """Control batch behavior with RunnableConfig."""
    print("\n⚙️ PART B: Batch Configuration")
    print("-" * 40)

    chain = create_chain()

    texts = ["Love it!", "Hate it!", "Okay.", "Great!", "Bad."]

    inputs = [{"text:t"} for t in texts ]

    # Configure batch processing
    config = RunnableConfig(
        max_concurrency=3, # Max 3 parallel requests
        tags=["sentiment-batch"], # For debugging/tracing
        metadata = {"batch_id": "demo-001"}  # Custom metadata
    )

    print(f"Config: max_concurrency=3, {len(inputs)} inputs")

    start = time.time()
    results = chain.batch(inputs, config=config)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s\n")
    print("Results:")
    for text, result in zip(texts, results):
        print(f"  {result.sentiment.value:8} | {text}")

# PART C: Error Handling in Batch
def demo_batch_errors():
    """Handle errors without failing the entire batch."""
    print("\n PART C: Error Handling in Batch")
    print("-" * 40)
    
    chain = create_chain()

    # Include some problematic inputs

    inputs = [
        {"text": "I love this!"},
        {"text": ""},  # Empty - might cause issues
        {"text": "Great product!"},
        {"text": "   "},  # Whitespace only
        {"text": "Terrible!"},
    ]

    # return_exceptions=True: Don't fail entire batch on one error
    results = chain.batch(inputs, return_exceptions=True)

    print("Results:")
    for i, (inp, result) in enumerate(zip(inputs, results)):
        text = inp["text"] or "(empty)"
        if isinstance(result, Exception):
            print(f"  {i+1}. ❌ ERROR: {type(result).__name__}")
        else:
            print(f"  {i+1}. ✅ {result.sentiment.value:8} | '{text[:20]}'")
    
    # Count successes
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    print(f"\nSuccess rate: {success_count}/{len(results)}")

# PART D: Async Processing
async def demo_async():
    """Use async for even better performance in async apps."""
    print("\n⚡ PART D: Async Processing")
    print("-" * 40)
    
    chain = create_chain()
    
    texts = ["Love it!", "Hate it!", "Meh."]
    
    # Single async call
    print("\n1. ainvoke() - Single async:")
    result = await chain.ainvoke({"text": texts[0]})
    print(f"   {texts[0]} → {result.sentiment.value}")
    
    # Async batch
    print("\n2. abatch() - Async batch:")
    inputs = [{"text": t} for t in texts]
    results = await chain.abatch(inputs)
    for text, result in zip(texts, results):
        print(f"   {text} → {result.sentiment.value}")
    
    # Concurrent tasks (advanced)
    print("\n3. Concurrent tasks with gather():")
    tasks = [chain.ainvoke({"text": t}) for t in texts]
    results = await asyncio.gather(*tasks)
    print(f"   All {len(results)} completed concurrently!")


# PART E: Practical Batch Processing Pattern
def demo_practical_pattern():
    """Real-world pattern for batch processing."""
    print("\n📦 PART E: Practical Batch Pattern")
    print("-" * 40)

    chain = create_chain()

    # Simulate a dataset
    dataset = [
        "This product exceeded my expectations!",
        "Worst purchase I've ever made.",
        "It works as described. Nothing special.",
        "Absolutely fantastic! Highly recommend!",
        "Complete waste of money. Avoid!",
        "Decent quality for the price.",
        "Life-changing product. 5 stars!",
        "Broke after one week. Disappointed.",
    ]

    print(f"Processing {len(dataset)} items...")

    # Prepare inputs
    inputs = [{"text:t"} for t in dataset]

    # Process with error handling
    results = chain.batch(
        inputs,
        config=RunnableConfig(max_concurrency=5),
        return_exceptions=True
    )
    
    # Analyze results
    successful = []
    failed = []

    for text, result in zip(dataset, results):
        if isinstance(result, Exception):
            failed.append({"text": text, "error": str(result)})
        else:
            successful.append({
                "text": text,
                "sentiment": result.sentiment.value,
                "confidence": result.confidence
            })
    
    # Summary
    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    # Sentiment distribution
    from collections import Counter
    sentiments = Counter(r["sentiment"] for r in successful)
    print(f"\nDistribution:")
    for sentiment, count in sentiments.items():
        bar = "█" * count
        print(f"  {sentiment:8} {bar} ({count})")

# TEST IT OUT
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: Understanding Batch Processing")
    print("=" * 60)
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set!")
        print("   Run: export GOOGLE_API_KEY='your-key-here'")
        print("\n   Showing concepts without running LLM calls...")
        
        print("\n" + "=" * 60)
        print("CONCEPT EXPLANATION:")
        print("=" * 60)
        print("""
# BATCH PROCESSING

# ❌ SLOW: Sequential loop
for text in texts:
    result = chain.invoke({"text": text})  # One at a time

# ✅ FAST: Batch (parallel)
inputs = [{"text": t} for t in texts]
results = chain.batch(inputs)  # All at once!

# With configuration
config = RunnableConfig(
    max_concurrency=5,  # Limit parallel requests
    tags=["my-batch"]   # For debugging
)
results = chain.batch(inputs, config=config)

# Handle errors gracefully
results = chain.batch(inputs, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        print("Failed!")
    else:
        print(result.sentiment)

# ASYNC (for async apps like FastAPI)
result = await chain.ainvoke({"text": "hello"})
results = await chain.abatch(inputs)
        """)
        
    else:
        demo_sequential_vs_batch()
        demo_batch_config()
        demo_batch_errors()
        
        # Run async demo
        print("\n" + "-" * 60)
        asyncio.run(demo_async())
        
        demo_practical_pattern()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Use .batch() instead of for loops - much faster!")
    print("2. RunnableConfig controls max_concurrency")
    print("3. return_exceptions=True prevents one failure from stopping all")
    print("4. Use .ainvoke() and .abatch() for async applications")
    print("5. Always handle both success and exception results")
    print("6. Batch is essential for production workloads")

    

