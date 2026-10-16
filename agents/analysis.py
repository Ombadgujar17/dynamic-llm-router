class AnalysisAgent:
    def __init__(self, router, providers):
        self.router, self.providers = router, providers

    def run(self, profile, task: str, plan: str) -> dict:
        prompt = f"You are the Analysis Agent. Follow this plan: {plan}\nTask: {task}\nProvide a clear, concise answer. Do not invent facts."
        return self.router.generate(profile, prompt, self.providers)
