"""
STEP 2: ChatPromptTemplate
==========================

CONCEPT: ChatPromptTemplate structures your prompts with roles.

WHY THIS MATTERS:
- LLMs work with message roles: system, human, assistant
- System message sets behavior/personality
- Human message contains the actual request
- Placeholders {like_this} get filled at runtime

KEY INSIGHT: Good prompts = Good outputs!
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# PART A: Basic ChatPromptTemplate
# Method 1: from_messages (most common)
basic_prompt = ChatPromptTemplate.format_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

# Method 2: Explicit message types (same result)
explicit_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    ("human", "{user_input}")  # Can mix styles
])

# PART B: Multiple Placeholders
sentiment_prompt  = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert sentiment analyzer.
Your task is to analyze the sentiment of {language} text.
Be {strictness} in your analysis.
"""),
(
    "human",
     """Analyze this text:
{text}
Provide sentiment, confidence, and key phrases."""
)
])

# PART C: Full Sentiment Analysis Prompt
FULL_SENTIMENT_PROMPT  = ChatPromptTemplate.from_messages([
   ( "system",
    """You are an expert sentiment analysis system with deep understanding
of language nuances, context, and emotional intelligence.

ANALYSIS GUIDELINES:
1. Consider the overall tone and specific word choices
2. Identify both explicit and implicit sentiment signals
3. Account for sarcasm, irony, and cultural context
4. Weight recent/final statements more heavily

CONFIDENCE SCORING:
- 0.95-1.0: Extremely clear, unambiguous sentiment
- 0.85-0.95: Clear sentiment with minor ambiguity
- 0.70-0.85: Moderate confidence, some mixed signals
- 0.50-0.70: Low confidence, genuinely ambiguous

Always extract exact phrases from the original text.
"""),(
        "human",
        """Analyze the sentiment of the following text:

---
{text}
---

Provide complete analysis with sentiment, confidence, emotions, 
key phrases, and summary."""
)
])

# PART D: Conversation History (for chat apps)
chat_prompt = ChatPromptTemplate.format_messages([
    ("system", "you are helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# TEST IT OUT
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: Understanding ChatPromptTemplate")
    print("=" * 60)
    
    # PART A: Basic prompt
    print("\n📝 PART A: Basic Prompt")
    print("-" * 40)
    result = basic_prompt.invoke({"user_input": "Hello, how are you?"})
    print("Input: {'user_input': 'Hello, how are you?'}")
    print("\nOutput messages:")
    for msg in result.messages:
        print(f"  [{msg.type:8}] {msg.content[:50]}...")
    
    # PART B: Multiple placeholders
    print("\n\n📝 PART B: Multiple Placeholders")
    print("-" * 40)
    result = sentiment_prompt.invoke({
        "language": "English",
        "strictness": "strict",
        "text": "I love this product!"
    })
    print("Input: {language, strictness, text}")
    print("\nOutput messages:")
    for msg in result.messages:
        print(f"  [{msg.type:8}] {msg.content[:60]}...")
    
    # PART C: Full prompt
    print("\n\n📝 PART C: Full Sentiment Prompt")
    print("-" * 40)
    result = FULL_SENTIMENT_PROMPT.invoke({
        "text": "The product quality is amazing but shipping was slow."
    })
    print("Formatted system message (first 200 chars):")
    print(f"  {result.messages[0].content[:200]}...")
    print("\nFormatted human message:")
    print(f"  {result.messages[1].content}")
    
    # PART D: With conversation history
    print("\n\n📝 PART D: With Conversation History")
    print("-" * 40)
    history = [
        HumanMessage(content="Hi, I need help with Python"),
        AIMessage(content="Of course! What do you need help with?"),
    ]
    result = chat_prompt.invoke({
        "history": history,
        "user_input": "How do I read a file?"
    })
    print("Messages in prompt:")
    for msg in result.messages:
        role = msg.type
        content = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
        print(f"  [{role:8}] {content}")
    
    # Show input variables
    print("\n\n📋 Prompt Input Variables:")
    print("-" * 40)
    print(f"  basic_prompt: {basic_prompt.input_variables}")
    print(f"  sentiment_prompt: {sentiment_prompt.input_variables}")
    print(f"  FULL_SENTIMENT_PROMPT: {FULL_SENTIMENT_PROMPT.input_variables}")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("1. ChatPromptTemplate.from_messages([...]) is the standard way")
    print("2. Roles: 'system' (behavior), 'human' (request), 'assistant' (AI)")
    print("3. Use {placeholders} for dynamic content")
    print("4. .invoke({'key': 'value'}) fills in placeholders")
    print("5. MessagesPlaceholder for conversation history")
    print("6. System prompt = where you put all the instructions")

