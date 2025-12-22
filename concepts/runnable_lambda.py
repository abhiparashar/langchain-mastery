"""
STEP 6: RunnableLambda
======================

CONCEPT: Wrap any Python function to use in LCEL chains.

WHY THIS MATTERS:
- Add preprocessing before the LLM
- Add postprocessing after the LLM
- Add error handling, logging, validation
- Custom transformations anywhere in the chain

KEY INSIGHT: RunnableLambda makes ANY function chainable with |
"""

import os
from enum import Enum
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# PART A: Basic RunnableLambda
def demo_basic_lambda():
    """The simplest RunnableLambda example."""
    print("\n🔧 PART A: Basic RunnableLambda")
    print("-" * 40)

    # A regular Python function
    def add_exclamation(text:str)->str:
        return text + "!!"
    
    # Wrap it to make it chainable
    add_exclamation_runnable = RunnableLambda(add_exclamation)

    # Now it has .invoke()!
    result = add_exclamation_runnable.invoke("HELLO")
    print(f"Input: 'Hello'")
    print(f"Output: '{result}'")

    # It can also batch!
    results = add_exclamation_runnable.batch(["Hi", "Bye", "Wow"])
    print(f"\nBatch: {results}")

# PART B: Preprocessing Chain
def demo_preprocessing():
    """Add preprocessing before the LLM."""
    print("\n🔧 PART B: Preprocessing")
    print("-" * 40)

    # Preprocessing function
    def clean_text(text:str)->str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        cleaned = " ".join(text.split)

        # Truncate if too long
        if len(cleaned)>1000:
            cleaned = cleaned[:1000] + "..."
        return cleaned
    
    # Wrap it
    preprocess  = RunnableLambda(clean_text)

    # Test
    messy_text = "   Hello    world!   \n\n  How   are   you?   "
    cleaned = preprocess.invoke(messy_text)

    print(f"Before: '{messy_text}'")
    print(f"After:  '{cleaned}'")

# PART C: Postprocessing Chain
class SentimentResult(BaseModel):
    sentiment:str = Field(description="positive, negative, or neutral")
    confidence:float = Field(le=0.0, ge=1.0)

def demo_postprocessing():
    """Add postprocessing after the LLM."""
    print("\n🔧 PART C: Postprocessing")
    print("-" * 40)

    # Postprocessing function
    def format_for_api(result:SentimentResult) -> dict:
        """Format result for API response."""
        return {
            "sentiment" :  result.sentiment,
            "confidence": round(result.confidence, 2),
            "confidence_percent": f"{result.confidence:.0%}",
            "is_positive": result.sentiment == "positive",
            "is_negative": result.sentiment == "negative",
        }
    
    postprocess  = RunnableLambda(format_for_api)

    # Simulate an LLM result
    llm_result  = SentimentResult(sentiment="positive", confidence=0.87654)
    formatted  = postprocess.invoke(llm_result)

    print(f"Before: SentimentResult(sentiment='positive', confidence=0.87654)")
    print(f"After:  {formatted}")

# PART D: Full Chain with Pre and Post Processing
    """Complete chain: preprocess → prompt → model → postprocess"""
    print("\n🔧 PART D: Full Chain with Pre/Post Processing")
    print("-" * 40)

    # 1. Preprocessing
    def preprocess(text:str)->dict:
        cleaned = " ".join(text.split()).strip()
        return {"text":cleaned}

    # 2. Postprocessing
    def postprocess(result:SentimentResult) -> dict:
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        return{
            "emoji": emoji.get(result.sentiment, "?"),
            "sentiment" : emoji.get(result.confidence),
            "confidence": f"{result.confidence:.0%}",
        }
    
    # 3. Build the chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze sentiment. Return positive, negative, or neutral."),
        ("human", "{text}")
    ])

    model = init_chat_model(model="germini-2.5-flash", model_provider="google_genai", temperature=0.1)

    structured_model = model.with_structured_output(SentimentResult)

    # THE FULL CHAIN
    chain = RunnableLambda(preprocess) | prompt | structured_model | RunnableLambda(postprocess)

    # Test it
    messy_input = "   I   LOVE   this   product!!!   "
    result = chain.invoke(messy_input)
    
    print(f"Input: '{messy_input}'")
    print(f"Output: {result}")
    
    return chain

# PART E: Error Handling Wrapper
def demo_error_handling():
    """Wrap a chain with error handling."""
    print("\n🔧 PART E: Error Handling Wrapper")
    print("-" * 40)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze sentiment."),
        ("human", "{text}")
    ])

    model = init_chat_model(model="geimi-2.5-flash", model_provider="google_genai")

    structured_model  = model.with_structured_output(SentimentResult)

    inner_chain = prompt | structured_model

    # Error handling wrapper
    def safe_analyze(text:str)->dict:
        try:
            if not text or not text.strip():
                return {"success": False, "error": "Empty input"}
            
            result = inner_chain.invoke({"text": text})

            return {
                "success": True,
                "result": {
                    "sentiment": result.sentiment,
                    "confidence": result.confidence
                },
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
        
    safe_chain = RunnableLambda(safe_analyze)

    # Test with various inputs
    test_cases = [
        "I love this!",
        "",
        "Great product!",
    ]
    
    print("Results:")
    for text in test_cases:
        result = safe_chain.invoke(text)
        status = "✅" if result["success"] else "❌"
        print(f"  {status} '{text or '(empty)'}': {result}")

# PART F: RunnablePassthrough (Pass Data Through)
def demo_passthrough():
    """Pass original data alongside transformed data."""
    print("\n🔧 PART F: RunnablePassthrough")
    print("-" * 40)

    from langchain_core.runnables import RunnableParallel

    # Sometimes you want to keep original data AND add new data
    def add_metadata(data: dict) -> dict:
        return {
            **data,
            "word_count": len(data["text"].split()),
            "char_count": len(data["text"])
        }
    
    # RunnablePassthrough passes input unchanged
    # Useful in RunnableParallel
    
    chain = RunnableLambda(add_metadata)
    
    result = chain.invoke({"text": "Hello world how are you"})
    print(f"Input: {{'text': 'Hello world how are you'}}")
    print(f"Output: {result}")

# PART G: Lambda with Logging
def demo_logging():
    """Add logging to see what's happening in your chain."""
    print("\n🔧 PART G: Lambda with Logging")
    print("-" * 40)
    
    def log_input(data):
        print(f"  📥 Input received: {str(data)[:50]}...")
        return data
    
    def log_output(data):
        print(f"  📤 Output produced: {str(data)[:50]}...")
        return data
    
    # Add logging to any chain
    def process(x):
        return x.upper()
    
    chain = (
        RunnableLambda(log_input)
        | RunnableLambda(process)
        | RunnableLambda(log_output)
    )
    
    print("Running chain...")
    result = chain.invoke("hello world")
    print(f"Final result: {result}")

# TEST IT OUT
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: Understanding RunnableLambda")
    print("=" * 60)
    
    # Parts A, B, C don't need API
    demo_basic_lambda()
    demo_preprocessing()
    demo_postprocessing()
    demo_passthrough()
    demo_logging()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set for LLM examples!")
        print("   Parts D and E require API key.")
        print("\n" + "=" * 60)
        print("CONCEPT EXPLANATION:")
        print("=" * 60)
        print("""
# RunnableLambda wraps any function for use in chains

# Basic usage
def my_func(x):
    return x.upper()

runnable = RunnableLambda(my_func)
result = runnable.invoke("hello")  # "HELLO"

# Use in chains
chain = (
    RunnableLambda(preprocess)    # Clean input
    | prompt                       # Create prompt
    | model                        # Call LLM
    | RunnableLambda(postprocess)  # Format output
)

# Common patterns:
# 1. Preprocessing: Clean/validate input before LLM
# 2. Postprocessing: Format/transform output after LLM
# 3. Error handling: Wrap chain in try/except
# 4. Logging: Add visibility into chain execution
        """)
    else:
        # demo_full_chain()
        demo_error_handling()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. RunnableLambda(func) makes any function chainable")
    print("2. Use for preprocessing (clean input before LLM)")
    print("3. Use for postprocessing (format output after LLM)")
    print("4. Use for error handling (wrap risky operations)")
    print("5. Use for logging/debugging (see what flows through)")
    print("6. Can be used anywhere in a chain with |")

    

