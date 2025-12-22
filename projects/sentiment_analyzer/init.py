"""
Sentiment Analyzer - LangChain 1.0 Project
===========================================

A production-ready sentiment analyzer demonstrating:
- Pydantic v2 schemas (Step 1)
- ChatPromptTemplate (Step 2)
- with_structured_output (Step 3)
- LCEL chains (Step 4)
- Batch processing (Step 5)
- RunnableLambda (Step 6)

Quick Start:
    from sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("I love this!")
    print(result.sentiment)  # positive
"""

from schema import (
    SentimentType,
    EmotionType,
    SentimentResult,
    AnalysisResponse,
)

from .prompts import (
    SENTIMENT_PROMPT,
    SYSTEM_PROMPT,
    HUMAN_PROMPT,
)

from .chain import create_chain

from .utils import (
    create_preprocessor,
    create_postprocessor,
    create_safe_chain,
    create_full_pipeline,
    format_result_display,
)

from .analyzer import SentimentAnalyzer


__all__ = [
    # Schemas
    "SentimentType",
    "EmotionType",
    "SentimentResult",
    "AnalysisResponse",
    # Prompts
    "SENTIMENT_PROMPT",
    "SYSTEM_PROMPT", 
    "HUMAN_PROMPT",
    # Chain
    "create_chain",
    # Utils
    "create_preprocessor",
    "create_postprocessor",
    "create_safe_chain",
    "create_full_pipeline",
    "format_result_display",
    # Main class
    "SentimentAnalyzer",
]

__version__ = "1.0.0"