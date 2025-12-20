MODELS = {
    # Google
    "gemini": "google_genai:gemini-2.5-flash",
    "gemini-pro": "google_genai:gemini-2.5-pro",
    
    # Anthropic
    "claude": "anthropic:claude-sonnet-4-20250514",
    "claude-haiku": "anthropic:claude-haiku-4-20250514",
    
    # OpenAI
    "gpt": "openai:gpt-4o",
    "gpt-mini": "openai:gpt-4o-mini"
}


DEFAULT_MODEL = "gemini"
DEFAULT_SESSION_ID = "default"

# Optional: Cost per 1M tokens (for estimation later)
COSTS = {
    "gemini": {"input": 0.075, "output": 0.30},
    "claude": {"input": 3.00, "output": 15.00},
    "gpt": {"input": 2.50, "output": 10.00},
}