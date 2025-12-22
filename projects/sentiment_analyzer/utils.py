"""
utils.py - Utility Functions with RunnableLambda
=================================================
Learned in: Step 6
"""

from langchain_core.runnables import RunnableLambda, Runnable

from .schema import SentimentResult, AnalysisResponse
from .chain import create_chain


def create_preprocessor() -> RunnableLambda:
    """
    Create text preprocessing step.
    
    Cleans and validates input text before sending to LLM.
    """
    def preprocess(text: str) -> dict:
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Clean whitespace
        cleaned = " ".join(text.split()).strip()
        
        if not cleaned:
            raise ValueError("Text cannot be only whitespace")
        
        # Truncate if too long
        if len(cleaned) > 10000:
            cleaned = cleaned[:10000] + "..."
        
        return {"text": cleaned}
    
    return RunnableLambda(preprocess)


def create_postprocessor() -> RunnableLambda:
    """
    Create result formatting step.
    
    Formats SentimentResult for API responses.
    """
    def postprocess(result: SentimentResult) -> dict:
        emoji_map = {
            "positive": "😊",
            "negative": "😞", 
            "neutral": "😐",
            "mixed": "🤔"
        }
        
        return {
            "sentiment": result.sentiment.value,
            "emoji": emoji_map.get(result.sentiment.value, "❓"),
            "confidence": round(result.confidence, 2),
            "confidence_percent": f"{result.confidence:.0%}",
            "emotions": [e.value for e in result.emotions],
            "key_phrases": result.key_phrases,
            "summary": result.summary,
            "is_positive": result.sentiment.value == "positive",
            "is_negative": result.sentiment.value == "negative",
        }
    
    return RunnableLambda(postprocess)


def create_safe_chain(
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None
) -> RunnableLambda:
    """
    Create error-handling wrapper around the chain.
    
    Returns AnalysisResponse instead of raising exceptions.
    """
    chain = create_chain(model_name=model_name, api_key=api_key)
    preprocessor = create_preprocessor()
    
    def safe_analyze(text: str) -> AnalysisResponse:
        try:
            # Preprocess
            cleaned = preprocessor.invoke(text)
            
            # Analyze
            result = chain.invoke(cleaned)
            
            return AnalysisResponse(
                success=True,
                result=result,
                error=None,
                metadata={
                    "input_length": len(text),
                    "word_count": len(text.split()),
                    "model": model_name
                }
            )
        except ValueError as e:
            return AnalysisResponse(
                success=False,
                result=None,
                error=f"Validation error: {e}"
            )
        except Exception as e:
            return AnalysisResponse(
                success=False,
                result=None,
                error=f"Analysis failed: {e}"
            )
    
    return RunnableLambda(safe_analyze)


def create_full_pipeline(
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None
) -> Runnable:
    """
    Create complete pipeline: preprocess → chain → postprocess
    
    Input: raw text string
    Output: formatted dict
    """
    chain = create_chain(model_name=model_name, api_key=api_key)
    
    pipeline = (
        create_preprocessor()
        | chain
        | create_postprocessor()
    )
    
    return pipeline


def format_result_display(result: SentimentResult) -> str:
    """Format result for console display."""
    emotions = ", ".join(e.value for e in result.emotions) or "none"
    phrases = "\n    - ".join(result.key_phrases) or "none"
    
    return f"""
┌────────────────────────────────────────
│ SENTIMENT: {result.sentiment.value.upper()}
│ CONFIDENCE: {result.confidence:.0%}
│ EMOTIONS: {emotions}
│ KEY PHRASES:
│   - {phrases}
│ SUMMARY: {result.summary}
└────────────────────────────────────────"""