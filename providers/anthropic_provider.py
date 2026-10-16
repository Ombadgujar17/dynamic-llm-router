import os, time
from providers.base import BaseProvider

class AnthropicProvider(BaseProvider):
    def __init__(self):
        from anthropic import Anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing from the local environment")
        self.client = Anthropic(api_key=key)

    def list_models(self) -> list[str]:
        # Lists models visible to this credential; this does not prove free access or task quality.
        response = self.client.models.list(limit=1000)
        return [item.id for item in response.data if getattr(item, "id", None)]

    def generate(self, model: str, prompt: str) -> dict:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=model, max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage_obj = getattr(response, "usage", None)
        usage = {"input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
                 "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0, "simulated": False}
        return {"text": text, "usage": usage, "latency_seconds": round(time.perf_counter() - start, 3), "simulated": False}
