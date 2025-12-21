"""
STEP 1: Pydantic Schemas
========================

CONCEPT: Pydantic models define the EXACT structure the LLM must return.

WHY THIS MATTERS:
- Without a schema, LLM returns raw text (unreliable to parse)
- With a schema, LLM returns a Python object (type-safe, validated)

KEY INSIGHT: Field descriptions become part of the prompt!
The more detailed your description, the more accurate the LLM output.
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator

# PART A: Simple Schema (Start Here)
class SimpleSentiment(BaseModel):
    """
    A basic sentiment result - just two fields.
    
    This is the simplest possible schema.
    """
    sentiment:str = Field(
        description = "Either 'positive', 'negative', or 'neutral'",
    )
    confidence:float = Field(
        description = "Confidence score from 0.0 to 1.0"
    )

# PART B: Using Enums for Type Safety
class SentimentType(str, Enum):
    """
    WHY ENUM?
    - Restricts values to only valid options
    - IDE autocomplete works
    - Prevents typos like "postive" or "Positive"
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class BetterSentiment(BaseModel):
    """Now sentiment MUST be one of the enum values."""
    
    sentiment: SentimentType = Field(
        description="The overall sentiment classification"
    )
    confidence: float = Field(
        ge=0.0,  # ge = greater than or equal
        le=1.0,  # le = less than or equal
        description="Confidence score from 0.0 to 1.0"
    )

# PART C: Full Schema with Detailed Descriptions
class EmotionType(str, Enum):
    """Emotions we want the LLM to detect."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    EXCITEMENT = "excitement"

class SentimentResult(BaseModel):
    """
    Complete sentiment analysis result.
    
    NOTICE: Each Field has a DETAILED description.
    These descriptions are sent to the LLM as instructions!
    
    Bad:  Field(description="The sentiment")
    Good: Field(description="The overall sentiment. Use 'positive' for 
          favorable opinions, 'negative' for unfavorable...")
    """
    sentiment: SentimentType = Field(
        description="The overall sentiment classification. "
                    "Use 'positive' for favorable opinions, "
                    "'negative' for unfavorable, "
                    "'neutral' for factual/objective, "
                    "'mixed' when both positive and negative are present."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0. "
                    "Use 0.9+ for very clear sentiment, "
                    "0.7-0.9 for fairly clear, "
                    "0.5-0.7 for ambiguous text."
    )

    emotions: list[EmotionType] = Field(
        default_factory=list,
        description="List of emotions detected, ordered by intensity. "
                    "Include only clearly expressed emotions (1-3 max)."
    )

    key_phrases:list[str] = Field(
        default_factory=list,
        description="2-5 exact phrases from the text that indicate sentiment."
    )

    summary: str = Field(
        description="1-2 sentence summary explaining WHY this sentiment."
    )

    # VALIDATORS: Add custom validation logic
    @field_validator('key_phrases')
    @classmethod
    def limit_phrases(cls, v: list[str]) -> list[str]:
        """Ensure max 5 phrases."""
        return v[:5] if len(v) > 5 else v


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Understanding Pydantic Schemas")
    print("=" * 60)

    # Create a valid result manually
    result = SentimentResult(
        sentiment=SentimentType.POSITIVE,
        confidence=0.95,
        emotions=[EmotionType.JOY, EmotionType.EXCITEMENT],
        key_phrases=["loved it", "amazing quality"],
        summary="Strong positive sentiment about product quality."
    )

    print("\n✅ Created SentimentResult:")
    print(f"   sentiment: {result.sentiment}")        # SentimentType.POSITIVE
    print(f"   sentiment.value: {result.sentiment.value}")  # "positive"
    print(f"   confidence: {result.confidence}")
    print(f"   emotions: {[e.value for e in result.emotions]}")
    print(f"   key_phrases: {result.key_phrases}")


    # Try invalid confidence (should fail)
    print("\n❌ Testing validation (confidence > 1.0):")
    try:
        bad_result = SentimentResult(
            sentiment=SentimentType.POSITIVE,
            confidence=1.5,  # Invalid! Must be <= 1.0
            emotions=[],
            key_phrases=[],
            summary="Test"
        )
    except Exception as e:
        print(f"Caught error: {type(e).__name__}")
        print(f"Validation works! ✓")
    
    # Show the JSON schema (what gets sent to LLM)
    print("\n📋 JSON Schema (sent to LLM):")
    import json
    schema = SentimentResult.model_json_schema()
    print(json.dumps(schema, indent=2)[:500] + "...")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. Pydantic models define the structure LLM must return")
    print("2. Field descriptions guide the LLM (be detailed!)")
    print("3. Enums restrict values to valid options")
    print("4. Validators add custom rules (ge, le, max length, etc.)")
    print("5. The JSON schema is sent to the LLM automatically")

    
