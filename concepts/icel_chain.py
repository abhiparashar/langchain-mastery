"""
STEP 4: LCEL Chain (LangChain Expression Language)
===================================================

CONCEPT: Chain components together with the pipe operator |

WHY THIS MATTERS:
- Clean, readable code: prompt | model | parser
- Automatic data flow between components
- Built-in async, batch, and streaming support

KEY INSIGHT: Each component's output becomes the next component's input.

    Input → Prompt → Model → Output
    {"text": "..."} → Messages → LLM Response → Parsed Result

NOTE: Requires GOOGLE_API_KEY to run LLM examples.
"""

import os
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# PART A: The Simplest Chain
def demo_simple_chain():
    """The most basic chain: prompt | model"""
    print("\n🔗 PART A: Simple Chain (prompt | model)")
    print("-" * 40)
    
    # Component 1: Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "you are a helpful assistant"),
        ("human", "{question}")
    ])

    # Component 2: Model
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    # THE MAGIC: Chain them with |
    chain = prompt | model

    # Invoke the chain
    result= chain.invoke({"question": "What is 2 + 2?"})

    print(f"Chain: prompt | model")
    print(f"Input: {{'question': 'What is 2 + 2?'}}")
    print(f"Output type: {type(result).__name__}")
    print(f"Output: {result.content[:100]}...")
    
    return chain

# PART B: Chain with Output Parser
def demo_chain_with_parser():
    """Add an output parser: prompt | model | parser"""
    print("\n🔗 PART B: Chain with Parser (prompt | model | parser)")
    print("-" * 40)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Translate to French."),
        ("human", "{question}")
    ])
    
    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.1)

    # StrOutputParser extracts just the string content
    parser = StrOutputParser()

    # Three components chained
    chain = prompt | model | parser

    result = chain.invoke({"text": "Hello, how are you?"})

    print(f"Chain: prompt | model | StrOutputParser()")
    print(f"Output type: {type(result).__name__}")  # str, not AIMessage!
    print(f"Output: {result}")
    
    return chain

# PART C: Chain with Structured Output (Sentiment Analyzer!)
class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class SentimentResult(BaseModel):
    sentiment:SentimentType = Field(
        description="The overall sentiment"
    )
    confidence:float = Field(
        le=0.0,
        ge=1.0,
        description="Confidence 0-1"
    )
    key_phrases: list[str] = Field(
        description="Key phrases from text"
    )
    summary:str = Field(
        description="Brief explanation"
    )

def demo_structured_chain():
    """The real deal: prompt | structured_model"""
    print("\n🔗 PART C: Structured Chain (prompt | structured_model)")
    print("-" * 40)

    # Component 1: Detailed prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
"""You are a sentiment analyzer. Analyze the sentiment of text.
            
GUIDELINES:
- positive: favorable, happy, satisfied
- negative: unfavorable, angry, disappointed  
- neutral: factual, objective
- mixed: both positive and negative present

Be precise with confidence scores.
"""),
("human", "Analyze: {text}")
    ])

    # Component 2: Model with structured output
    model = init_chat_model(model="gemini-2.5-flash", model_provider="ggoogle_genai", temperature = 0.5)

    structured_llm = model.with_structured_output(SentimentResult)

    # Chain them!
    chain = prompt | structured_llm

    # Test it
    test_texts = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible experience. Waste of money. Never again.",
        "The package arrived on Tuesday. It was a box.",
        "Great food but awful service. Mixed feelings.",
    ]
    
    print("\nResults:")
    for text in test_texts:
        result = chain.invoke({"text": text})
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "🤔"}
        e = emoji.get(result.sentiment.value, "❓")
        print(f"  {e} {result.sentiment.value:8} ({result.confidence:.0%}) | {text[:40]}...")
    
    return chain


# PART D: Understanding Data Flow
def demo_data_flow():
    """Visualize what happens at each step."""
    print("\n🔗 PART D: Understanding Data Flow")
    print("-" * 40)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Reply briefly."),
        ("human", "{question}")
    ])

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature =0.1)

    # Step by step (what the chain does internally)
    print("\nManual step-by-step:")

    # Step 1: Invoke prompt
    input_data = {"question": "What is Python?"}
    print(f"  1. Input: {input_data}")

    prompt_output = prompt.invoke(input_data)
    print(f"  2. After prompt: {type(prompt_output).__name__} with {len(prompt_output.messages)} messages")

    # Step 2: Invoke model
    model_output = model.invoke(prompt_output)
    print(f"  3. After model: {type(model_output).__name__}")
    print(f"     Content: {model_output.content[:50]}...")

    # Now with chain (same result, cleaner code!)
    print("\nWith chain (prompt | model):")
    chain = prompt | model
    chain_output = chain.invoke(input_data)
    print(f"  Result: {chain_output.content[:50]}...")

# PART E: Chain Methods
def demo_chain_methods():
    """Show different ways to call a chain."""
    print("\n🔗 PART E: Chain Methods (invoke, batch, stream)")
    print("-" * 40)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Be very brief. One sentence max."),
        ("human", "What is {topic}?")
    ])

    model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    chain = prompt | model | StrOutputParser()

    # Method 1: invoke (single input)
    print("\n1. invoke() - Single input:")
    result = chain.invoke({"topic": "Python"})
    print(f"   {result[:60]}...")

     # Method 2: batch (multiple inputs, parallel!)
    print("\n2. batch() - Multiple inputs (parallel):")
    results = chain.batch([
        {"topic": "Python"},
        {"topic": "JavaScript"},
        {"topic": "Rust"}
    ])

    for topic, result in zip(["Python", "JavaScript", "Rust"], results):
        print(f"   {topic}: {result[:40]}...")

    # Method 3: stream (real-time output)
    print("\n3. stream() - Real-time tokens:")
    print("   ", end="")
    for chunk in chain.stream({"topic": "AI"}):
        print(chunk, end="", flush=True)
    print()

# TEST IT OUT
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: Understanding LCEL Chains")
    print("=" * 60)
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set!")
        print("   Run: export GOOGLE_API_KEY='your-key-here'")
        print("\n   Showing concepts without running LLM calls...")
        
        print("\n" + "=" * 60)
        print("CONCEPT EXPLANATION:")
        print("=" * 60)
        print("""
# LCEL uses the pipe operator | to chain components

# Simple chain
chain = prompt | model

# With output parser
chain = prompt | model | StrOutputParser()

# With structured output
structured_model = model.with_structured_output(MySchema)
chain = prompt | structured_model

# Calling the chain
result = chain.invoke({"key": "value"})      # Single
results = chain.batch([{...}, {...}])         # Parallel
for chunk in chain.stream({...}):             # Real-time
    print(chunk)

# Data flows through each component:
# {"text": "hello"} → prompt → messages → model → response → parser → str
        """)
        
    else:
        demo_simple_chain()
        demo_chain_with_parser()
        demo_structured_chain()
        demo_data_flow()
        demo_chain_methods()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Use | to chain components: prompt | model | parser")
    print("2. Data flows left to right through each component")
    print("3. .invoke() for single inputs")
    print("4. .batch() for parallel processing (faster!)")
    print("5. .stream() for real-time output")
    print("6. with_structured_output() returns Pydantic objects")
    print("7. Chains are reusable - define once, call many times")




