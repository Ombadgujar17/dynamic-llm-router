
from dataclasses import replace

from router.policies import classify_task
from agents.analysis import AnalysisAgent
from agents.verification import VerificationAgent


class CoordinatorAgent:
    def __init__(self, router, providers):
        self.router = router
        self.providers = providers

    def run(self, task: str, sensitivity: str = "public") -> dict:
        profile = classify_task(task, sensitivity)

        # Planning is usually simpler than solving the original task.
        # Keep the task type and sensitivity, but lower planning requirements.
        planning_profile = replace(
            profile,
            complexity=2,
            quality_required=3,
            high_impact=False,
        )

        plan = self.router.generate(
            planning_profile,
            f"Create a concise execution plan for this task. "
            f"Return 2-4 steps. Task: {task}",
            self.providers,
        )

        analysis = AnalysisAgent(self.router, self.providers).run(
            profile, task, plan["text"]
        )

        verification = VerificationAgent(self.router, self.providers).run(
            profile, task, analysis["text"]
        )

        passed = verification["passed"]
        escalation = None

        # One bounded escalation attempt if verification fails.
        if not passed and self.router.calls < self.router.max_calls:
            avoid = {f'{analysis["provider"]}:{analysis["model"]}'}
            try:
                escalation = self.router.generate(
                    profile,
                    f"Improve this draft using the verification findings. "
                    f"Task: {task}\nDraft: {analysis['text']}\n"
                    f"Findings: {verification['issues']}",
                    self.providers,
                    avoid=avoid,
                )
            except Exception as exc:
                self.router.trace.append({
                    "event": "escalation_unavailable",
                    "reason": type(exc).__name__,
                })

        final_answer = escalation["text"] if escalation else analysis["text"]

        return {
            "task_profile": profile.__dict__,
            "planning_profile": planning_profile.__dict__,
            "plan": plan,
            "analysis": analysis,
            "verification": verification,
            "escalation": escalation,
            "final_answer": final_answer,
            "trace": self.router.trace,
            "mode": self.router.mode,
            "session_spend_usd": self.router.session_spend,
        }
