#!/usr/bin/env python3
"""
main.py - Sentiment Analyzer Examples
======================================

Demonstrates all features learned in Steps 1-6.

Usage:
    export GOOGLE_API_KEY="your-key"
    python main.py
"""

import os
import sys
import asyncio

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    create_safe_chain,
    create_full_pipeline,
)


def example_1_basic():
    """Basic single analysis."""
    print("\n" + "=" * 60)
    print("📝 EXAMPLE 1: Basic Analysis")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible experience. Complete waste of money.",
        "The package arrived. It was a box.",
        "Great food but awful service. Mixed feelings.",
    ]
    
    for text in texts:
        result = analyzer.analyze(text)
        analyzer.display(result)


def example_2_batch():
    """Batch processing multiple texts."""
    print("\n" + "=" * 60)
    print("📦 EXAMPLE 2: Batch Processing")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    texts = [
        "Love it! ⭐⭐⭐⭐⭐",
        "Meh, it's okay.",
        "DO NOT BUY!!!",
        "Works as expected.",
        "Changed my life!",
    ]
    
    print(f"Processing {len(texts)} texts in parallel...\n")
    
    results = analyzer.analyze_batch(texts, max_concurrency=3)
    
    emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "🤔"}
    
    for text, result in zip(texts, results):
        if result:
            e = emoji_map.get(result.sentiment.value, "❓")
            print(f"  {e} {result.sentiment.value:8} ({result.confidence:.0%}) | {text}")
        else:
            print(f"  ❌ FAILED | {text}")


def example_3_safe():
    """Error handling with safe chain."""
    print("\n" + "=" * 60)
    print("🛡️ EXAMPLE 3: Safe Analysis (Error Handling)")
    print("=" * 60)
    
    safe_chain = create_safe_chain()
    
    test_cases = [
        ("Great product!", "Valid text"),
        ("", "Empty string"),
        ("   ", "Whitespace only"),
    ]
    
    for text, desc in test_cases:
        result = safe_chain.invoke(text)
        status = "✅" if result.success else "❌"
        if result.success:
            print(f"  {status} {desc}: {result.result.sentiment.value}")
        else:
            print(f"  {status} {desc}: {result.error}")


def example_4_pipeline():
    """Full pipeline with pre/post processing."""
    print("\n" + "=" * 60)
    print("🔄 EXAMPLE 4: Full Pipeline")
    print("=" * 60)
    
    pipeline = create_full_pipeline()
    
    # Messy input
    messy = "   I    REALLY   love   this!!!   \n\n  Best ever!   "
    
    print(f"Input: '{messy}'")
    result = pipeline.invoke(messy)
    print(f"\nOutput: {result}")


async def example_5_async():
    """Async processing."""
    print("\n" + "=" * 60)
    print("⚡ EXAMPLE 5: Async Processing")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # Single async
    result = await analyzer.analyze_async("I love async Python!")
    print(f"  Single async: {result.sentiment.value}")
    
    # Batch async
    texts = ["Great!", "Bad!", "Okay."]
    results = await analyzer.analyze_batch_async(texts)
    print(f"  Batch async: {[r.sentiment.value if r else 'FAIL' for r in results]}")


def main():
    print("=" * 60)
    print("🎯 SENTIMENT ANALYZER - FINAL PROJECT")
    print("=" * 60)
    print("\nThis combines all concepts from Steps 1-6:")
    print("  • Step 1: Pydantic Schemas")
    print("  • Step 2: ChatPromptTemplate")
    print("  • Step 3: with_structured_output()")
    print("  • Step 4: LCEL Chains")
    print("  • Step 5: Batch Processing")
    print("  • Step 6: RunnableLambda")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  Set GOOGLE_API_KEY to run examples!")
        print("   export GOOGLE_API_KEY='your-key-here'")
        return
    
    example_1_basic()
    example_2_batch()
    example_3_safe()
    example_4_pipeline()
    asyncio.run(example_5_async())
    
    print("\n" + "=" * 60)
    print("✅ ALL EXAMPLES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()