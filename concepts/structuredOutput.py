"""
STEP 3: with_structured_output()
================================

CONCEPT: Force the LLM to return a Pydantic object instead of raw text.

WHY THIS MATTERS:
- Without: LLM returns "The sentiment is positive with 95% confidence..."
- With:    LLM returns SentimentResult(sentiment="positive", confidence=0.95)

KEY INSIGHT: No more regex parsing! Get Python objects directly.

NOTE: This step requires an API key to run.
      Set GOOGLE_API_KEY environment variable.
"""

import os
from enum import Enum
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# PART A: Define Our Schema (from Step 1)
class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class SentimentResult(BaseModel):
    """The structure we want the LLM to return."""
    sentiment: SentimentType = Field(
        description="The overall sentiment: positive, negative, neutral, or mixed"
    )

    confidence:float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    key_phrases:list[str] = Field(
        description="2-3 phrases from the text that indicate the sentiment"
    )
    summary:str = Field(
        description="One sentence explaining why this sentiment was detected"
    )

# PART B: Create LLM WITHOUT Structured Output
def demo_without_structured():
    """Show what happens without structured output."""
    print("\n WITHOUT structured output:")
    print("-" * 40)

    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature = 0.7)

    response = llm.invoke("Analyze the sentiment of: 'I love this product! Best purchase ever!'")

    print(f"Type: {type(response)}")
    print(f"Content type: {type(response.content)}")
    print(f"Content: {response.content[:200]}...")
    print("\n Problem: It's just a string! You'd need regex to parse it.")

# PART C: Create LLM WITH Structured Output
def demo_with_structured():
    """Show the magic of structured output."""
    print("\n✅ WITH structured output:")
    print("-" * 40)

    llm  = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature = 0.1)

    # THE MAGIC: .with_structured_output(PydanticModel)
    structured_llm = llm.with_structured_output(SentimentResult)

    response = structured_llm.invoke( "Analyze the sentiment of: 'I love this product! Best purchase ever!'")

    print(f"Type: {type(response)}")
    print(f"Is SentimentResult? {isinstance(response, SentimentResult)}")
    print()
    print("Accessing fields directly:")
    print(f"  response.sentiment = {response.sentiment}")
    print(f"  response.sentiment.value = {response.sentiment.value}")
    print(f"  response.confidence = {response.confidence}")
    print(f"  response.key_phrases = {response.key_phrases}")
    print(f"  response.summary = {response.summary}")
    
    return response

# PART D: Different Schemas for Different Tasks
class PersonInfo(BaseModel):
    """Extract person information from text."""
    name:str = Field(description="Person's full name")
    age:int | None = Field(description="Person's age if mentioned")
    occupation : str | None = Field(description="Person's job if mentioned")

class ProductReview(BaseModel):
    """Extract review information."""
    product_name: str = Field(description="Name of the product")
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    pros: list[str] = Field(description="Positive aspects mentioned")
    cons: list[str] = Field(description="Negative aspects mentioned")

def demo_multiple_schemas():
    """Show how to use different schemas for different tasks."""
    print("\n Multiple schemas for different tasks:")
    print("-" * 40)
    
    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature = 0.1)

    # Schema 1: Person extraction

    person_llm = llm.with_structured_output(PersonInfo)

    person = person_llm.invoke("John Smith is a 35-year-old software engineer from Seattle.")

    print(f"\nPersonInfo: name={person.name}, age={person.age}, job={person.occupation}")

    # Schema 2: Review extraction 
    product_review_llm = llm.with_structured_output(ProductReview)

    review = product_review_llm.invoke(
        "iPhone 15 Pro review: Amazing camera and fast processor. "
        "Battery life is great. However, it's expensive and heavy. 4 stars."
    )

    print(f"\nProductReview: {review.product_name}")
    print(f"  Rating: {'⭐' * review.rating}")
    print(f"  Pros: {review.pros}")
    print(f"  Cons: {review.cons}")

# PART E: Error Handling
def demo_error_handling():
    """Show how to handle structured output errors."""
    print("\n Error handling:")
    print("-" * 40)

    llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature = 0.1)

    structured_llm  = llm.with_structured_output(SentimentResult)

    test_cases = [
        ("I love this!", "Valid positive"),
        ("I hate this!", "Valid negative"),
        ("", "Empty string"),
    ]

    for text, description in test_cases:
        try:
            if text:
                result = structured_llm.invoke(f"Analyze sentiment: '{text}'")
                print(f"{description}: {result.sentiment.value}")

        except Exception as e:
            print(f"{description}: {type(e).__name__}")


# TEST IT OUT
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: Understanding with_structured_output()")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  GOOGLE_API_KEY not set!")
        print("   Run: export GOOGLE_API_KEY='your-key-here'")
        print("\n   Showing concepts without running LLM calls...")
        
        print("\n" + "=" * 60)
        print("CONCEPT EXPLANATION (no API needed):")
        print("=" * 60)
        print("""
# Step 1: Create the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Step 2: Wrap with structured output
structured_llm = llm.with_structured_output(SentimentResult)

# Step 3: Call it - returns Pydantic object, not string!
result = structured_llm.invoke("Analyze: I love this!")

# Step 4: Use typed fields directly
print(result.sentiment)      # SentimentType.POSITIVE
print(result.confidence)     # 0.95
print(result.key_phrases)    # ["love this"]
        """)
        
    else:
        print(f"\n✅ API key found: {os.environ['GOOGLE_API_KEY'][:10]}...")
        
        demo_without_structured()
        result = demo_with_structured()
        demo_multiple_schemas()
        demo_error_handling()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. .with_structured_output(Schema) wraps your LLM")
    print("2. Returns Pydantic objects, not strings")
    print("3. Fields are validated automatically")
    print("4. Access fields with dot notation: result.sentiment")
    print("5. Same LLM can use different schemas for different tasks")
    print("6. No more regex parsing!")
