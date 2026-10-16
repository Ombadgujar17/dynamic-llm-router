import os, time
from providers.base import BaseProvider

class GroqProvider(BaseProvider):
    def __init__(self):
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is missing from the local environment")
        self.client = Groq(api_key=key)

    def list_models(self) -> list[str]:
        # Model discovery does not generate text. Availability does not imply free access.
        response = self.client.models.list()
        return [item.id for item in response.data if getattr(item, "id", None)]

    def generate(self, model: str, prompt: str) -> dict:
        start = time.perf_counter()
        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        choice = response.choices[0]
        usage_obj = getattr(response, "usage", None)
        usage = {"input_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                 "output_tokens": getattr(usage_obj, "completion_tokens", 0) or 0, "simulated": False}
        return {"text": choice.message.content or "", "usage": usage, "latency_seconds": round(time.perf_counter() - start, 3), "simulated": False}
