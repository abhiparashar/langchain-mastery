from enum import Enum
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

class SentimentType(str, Enum):
    POSITIVE =  "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"

# Step-1 : Schema for output
class SentimentResult(BaseModel):
    sentiment:SentimentType = Field(description="sentiment can be positive, negative and mixed")
    confidence:float = Field(le=1.0, ge=0.0, description="confidence can be between 0 and 1")

# Step-2 : Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a sentiment analyzer"),
    ("human", "{text}")
])

# Step-3 : model
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
structured_model = model.with_structured_output(SentimentResult)


# Step-4 : preprocess definiton
def preprocess(text:str)->dict:
    cleaned = " ".join(text.split()).strip()
    return {"text": cleaned}

# step-5 : chaining
chain = RunnableLambda(preprocess) | prompt | structured_model

# step-5 : invoking
result = chain.invoke("Product was really really good")
print(result)

