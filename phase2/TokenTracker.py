from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

class TokenTracker:
    def __init__(self):
        self.callbacks = UsageMetadataCallbackHandler()

    def get_config(self)->dict:
        """Return config to pass to model calls."""
        return {"callbacks": self.callbacks}
    
    def get_usage(self)->dict:
        """Get usage breakdown by model."""
        return dict(self.callbacks.usage_metadata)
    
    def get_totals(self) -> dict:
        """Get total tokens across all models."""
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for model_usage in self.callbacks.usage_metadata.values:
            totals["input_tokens"] += model_usage.get("input_tokens", 0)
            totals["output_tokens"] += model_usage.get("output_tokens", 0)
            totals["total_tokens"] += model_usage.get("total_tokens", 0)
        return totals
    
    def print_usage(self):
        """Print formatted usage report."""
        print("\n=== Token Usage ===")
        for model, usage in self.callback.usage_metadata.items():
            print(f"\n{model}:")
            print(f"  Input:  {usage.get('input_tokens', 0):,}")
            print(f"  Output: {usage.get('output_tokens', 0):,}")
            print(f"  Total:  {usage.get('total_tokens', 0):,}")
        totals = self.get_totals()
        print(f"\n--- Grand Total ---")
        print(f"  {totals['total_tokens']:,} tokens")
