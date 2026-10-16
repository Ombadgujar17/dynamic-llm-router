import time
from providers.base import BaseProvider

class MockProvider(BaseProvider):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def generate(self, model: str, prompt: str) -> dict:
        start = time.perf_counter()
        # Intentionally deterministic demo output; never presented as a real model response.
        if prompt.startswith("Verification Agent:"):
            text = "MOCK VERIFICATION: PASS — the draft is present. This simulation does not establish factual correctness."
        elif prompt.startswith("Create a concise execution plan"):
            text = "MOCK PLAN: 1) identify the task 2) produce a draft 3) verify required properties."
        else:
            excerpt = prompt.strip().replace("\n", " ")[:220]
            text = f"MOCK RESPONSE ({self.provider_name}/{model}): Processed demo input: {excerpt}"
        return {"text": text, "usage": {"input_tokens": max(1, len(prompt.split())), "output_tokens": max(1, len(text.split())), "simulated": True}, "latency_seconds": round(time.perf_counter() - start, 4), "simulated": True}
