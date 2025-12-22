"""
analyzer.py - Main Sentiment Analyzer Class
============================================
Combines: All Steps (1-6)
- Schemas (Step 1)
- Prompts (Step 2)
- Structured Output (Step 3)
- LCEL Chain (Step 4)
- Batch Processing (Step 5)
- RunnableLambda utilities (Step 6)
"""

from langchain_core.runnables import RunnableConfig

from .schema import SentimentResult, AnalysisResponse
from .chain import create_chain
from .utils import create_preprocessor, format_result_display


class SentimentAnalyzer:
    """
    Production-ready Sentiment Analyzer.
    
    Example:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I love this product!")
        print(result.sentiment)  # positive
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        api_key: str | None = None
    ):
        self.model_name = model_name
        self.chain = create_chain(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        self.preprocessor = create_preprocessor()
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Uses: .invoke() (Step 4)
        """
        cleaned = self.preprocessor.invoke(text)
        return self.chain.invoke(cleaned)
    
    def analyze_safe(self, text: str) -> AnalysisResponse:
        """
        Analyze with error handling - never raises.
        
        Uses: RunnableLambda pattern (Step 6)
        """
        try:
            result = self.analyze(text)
            return AnalysisResponse(
                success=True,
                result=result,
                metadata={"model": self.model_name, "chars": len(text)}
            )
        except Exception as e:
            return AnalysisResponse(
                success=False,
                error=str(e)
            )
    
    def analyze_batch(
        self,
        texts: list[str],
        max_concurrency: int = 5
    ) -> list[SentimentResult | None]:
        """
        Analyze multiple texts in parallel.
        
        Uses: .batch() (Step 5)
        """
        # Preprocess all texts
        inputs = []
        for t in texts:
            try:
                inputs.append(self.preprocessor.invoke(t))
            except:
                inputs.append({"text": ""})  # Will fail gracefully
        
        # Batch process
        config = RunnableConfig(max_concurrency=max_concurrency)
        results = self.chain.batch(inputs, config=config, return_exceptions=True)
        
        # Convert exceptions to None
        return [None if isinstance(r, Exception) else r for r in results]
    
    async def analyze_async(self, text: str) -> SentimentResult:
        """
        Async analysis for async applications.
        
        Uses: .ainvoke() (Step 5)
        """
        cleaned = self.preprocessor.invoke(text)
        return await self.chain.ainvoke(cleaned)
    
    async def analyze_batch_async(
        self,
        texts: list[str],
        max_concurrency: int = 5
    ) -> list[SentimentResult | None]:
        """
        Async batch processing.
        
        Uses: .abatch() (Step 5)
        """
        inputs = []
        for t in texts:
            try:
                inputs.append(self.preprocessor.invoke(t))
            except:
                inputs.append({"text": ""})
        
        config = RunnableConfig(max_concurrency=max_concurrency)
        results = await self.chain.abatch(inputs, config=config, return_exceptions=True)
        
        return [None if isinstance(r, Exception) else r for r in results]
    
    def display(self, result: SentimentResult) -> None:
        """Print formatted result to console."""
        print(format_result_display(result))