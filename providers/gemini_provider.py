import os, time
from providers.base import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(self):
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing from the local environment")
        self.client = genai.Client(api_key=key)

    def list_models(self) -> list[str]:
        # Google returns model resource names; normalize to the ID expected by generate_content.
        model_ids = []
        for item in self.client.models.list():
            name = getattr(item, "name", None)
            if name:
                model_ids.append(name.removeprefix("models/"))
        return model_ids

    def generate(self, model: str, prompt: str) -> dict:
        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=model, contents=prompt,
            config={"max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))},
        )
        usage_obj = getattr(response, "usage_metadata", None)
        usage = {"input_tokens": getattr(usage_obj, "prompt_token_count", 0) or 0,
                 "output_tokens": getattr(usage_obj, "candidates_token_count", 0) or 0, "simulated": False}
        return {"text": getattr(response, "text", "") or "", "usage": usage,
                "latency_seconds": round(time.perf_counter() - start, 3), "simulated": False}
