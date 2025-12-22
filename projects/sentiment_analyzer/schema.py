from enum import Enum
from pydantic import BaseModel, Field, field_validator

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"

class SentimentResult(BaseModel):
    sentiment:SentimentType = Field(description="sentiment can be positivr, negative and mixed")
    confidence:float = Field(le=1.0, ge=0.0, description="confidence can be between 0 to 1")
    emotions:list[EmotionType] = Field(default_factory=list, description="List of detected emotions, ordered by intensity (1-3 max)")
    key_phrases : list[str] = Field(default_factory=list, description="2-5 exact phrases from text indicating sentiment")
    summary:str = Field(description="1-2 sentence explanation of the sentiment")

@field_validator
@classmethod
def limit_phrases(cls, v):
    v[:5] if len(v)>5 else v

@field_validator
@classmethod
def limit_emotions(cls, v):
    v[:3] if len(v)>3 else v

class AnalysisResponse(BaseModel):
    success:bool
    result : SentimentResult | None = None
    err: str | None = None
    metadata = dict | None = None

