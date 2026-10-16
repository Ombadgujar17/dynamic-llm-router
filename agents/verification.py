
from dataclasses import replace


class VerificationAgent:
    def __init__(self, router, providers):
        self.router = router
        self.providers = providers

    def run(self, profile, task: str, draft: str) -> dict:
        issues = []

        if not draft.strip():
            issues.append("Draft is empty")
        elif len(draft.strip()) < 20:
            issues.append("Draft may be too short to address the task")

        # Give the verifier its own routing profile.
        verification_profile = replace(
            profile,
            task_type="verification",
            complexity=max(2, profile.complexity),
            quality_required=max(3, profile.quality_required),
            high_impact=False,
        )

        check = self.router.generate(
            verification_profile,
            (
                "You are the Verification Agent. Review the draft against "
                "the task. Look for missing requirements, contradictions, "
                "and unsupported claims. Return PASS or FAIL followed by "
                "one short reason. Do not claim external facts were checked "
                "unless evidence is provided.\n"
                f"Task: {task}\nDraft: {draft}"
            ),
            self.providers,
        )

        response = check["text"].strip().lower()

        # Conservative prototype signal, not a factuality guarantee.
        if response.startswith("fail") or "issue" in response[:120]:
            issues.append("Model verifier flagged a potential issue")

        passed = len(issues) == 0

        return {
            "passed": passed,
            "status": (
                "checks_passed_factuality_unverified"
                if passed else "issues_found"
            ),
            "issues": issues,
            "structural_checks_passed": (
                not any(
                    issue in {"Draft is empty",
                              "Draft may be too short to address the task"}
                    for issue in issues
                )
            ),
            "factual_correctness_verified": False,
            "model_review_simulated": check["usage"].get(
                "simulated", False
            ),
            "model_review": check,
            "note": (
                "PASS means only that the prototype checks found no "
                "issues. It does not establish factual correctness."
            ),
        }
