import os, time
from providers.base import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing from the local environment")
        self.client = OpenAI(api_key=key)

    def list_models(self) -> list[str]:
        # Model discovery does not generate text or intentionally incur inference charges.
        response = self.client.models.list()
        return [item.id for item in response.data if getattr(item, "id", None)]

    def generate(self, model: str, prompt: str) -> dict:
        start = time.perf_counter()
        response = self.client.responses.create(model=model, input=prompt)
        usage_obj = getattr(response, "usage", None)
        usage = {"input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
                 "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0, "simulated": False}
        return {"text": response.output_text, "usage": usage, "latency_seconds": round(time.perf_counter() - start, 3), "simulated": False}
