from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    task: str
    task_type: str = "general"
    complexity: int = 2  # 1 simple ... 5 complex
    quality_required: int = 3  # 1 ... 5
    sensitivity: str = "public"  # public, internal, sensitive, restricted
    high_impact: bool = False
    max_latency_score: int = 5



def classify_task(task: str, sensitivity: str = "public") -> TaskProfile:
    """Explainable keyword classifier; not an ML classifier."""
    text = task.lower()

    # Check specific intents before broader keywords such as "python" or "code".
    if any(word in text for word in (
        "verify", "verification", "fact-check", "fact check",
        "validate", "evaluate", "review this answer",
        "check whether", "compare evidence"
    )):
        task_type, complexity, quality, impact = "verification", 4, 4, False

    elif any(word in text for word in (
        "healthcare", "medical", "diagnosis", "patient", "clinical"
    )):
        task_type, complexity, quality, impact = "reasoning", 4, 5, True

    elif any(word in text for word in (
        "code", "debug", "program", "function", "python"
    )):
        task_type, complexity, quality, impact = "coding", 4, 4, False

    elif any(word in text for word in ("classify", "label", "category")):
        task_type, complexity, quality, impact = "classification", 1, 3, False

    elif any(word in text for word in ("summarize", "summary", "summarise")):
        task_type, complexity, quality, impact = "summarization", 2, 3, False

    elif len(text.split()) > 70:
        task_type, complexity, quality, impact = "reasoning", 4, 4, False

    else:
        task_type, complexity, quality, impact = "general", 2, 3, False

    if sensitivity in {"sensitive", "restricted"}:
        quality = max(quality, 3)

    return TaskProfile(
        task, task_type, complexity, quality,
        sensitivity, impact
    )
