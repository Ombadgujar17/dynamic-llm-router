from dataclasses import asdict
from typing import Any
import os
from router.models import ModelSpec
from router.policies import TaskProfile


class RoutingBlocked(RuntimeError):
    pass


class Router:
    def __init__(self, registry: list[ModelSpec], mode: str = "mock", allow_paid: bool = False,
                 allow_verified_free: bool = False, max_calls: int = 5,
                 max_task_cost: float = 0.0, session_budget: float = 0.0):
        self.registry = registry
        self.mode = mode
        self.allow_paid = allow_paid
        self.allow_verified_free = allow_verified_free
        self.max_calls = max_calls
        self.max_task_cost = max_task_cost
        self.session_budget = session_budget
        self.calls = 0
        self.session_spend = 0.0
        self.trace: list[dict[str, Any]] = []

    def eligible_models(self, profile: TaskProfile, available_providers: set[str] | None = None) -> tuple[list[ModelSpec], list[dict[str, str]]]:
        eligible, excluded = [], []
        available_providers = available_providers if available_providers is not None else {m.provider for m in self.registry}
        for model in self.registry:
            reason = None
            if model.provider not in available_providers:
                reason = "Provider is not configured/available for this run"
            elif profile.sensitivity in {"sensitive", "restricted"} and model.provider not in {
                item.strip().lower() for item in os.getenv("SENSITIVE_DATA_ALLOWED_PROVIDERS", "openai").split(",") if item.strip()
            }:
                # Conservative default. Add providers only after reviewing data terms and organizational policy.
                reason = "Provider excluded by configured sensitive-data allowlist"
            elif profile.task_type != "general" and profile.task_type not in model.capabilities:
                reason = f"Model does not declare capability for task type '{profile.task_type}'"
            elif model.latency_score > profile.max_latency_score:
                reason = f"Latency score {model.latency_score} exceeds task limit {profile.max_latency_score}"
            elif model.quality_score < max(profile.quality_required, 4 if profile.complexity >= 4 else 1):
                reason = f"Quality score {model.quality_score} below requirement for complexity/quality"
            elif self.mode == "live" and not self.allow_paid:
                if not (self.allow_verified_free and model.free_tier_verified):
                    reason = "Live inference blocked: free eligibility is not verified and paid calls are disabled"
            elif self.mode == "live" and self.allow_paid and not model.pricing_known:
                reason = "Live inference blocked: model pricing is unknown; configure verified input/output prices first"
            if reason:
                excluded.append({"model": model.model_id, "provider": model.provider, "reason": reason})
            else:
                eligible.append(model)
        return eligible, excluded

    
    def select(
        self,
        profile: TaskProfile,
        avoid: set[str] | None = None,
        available_providers: set[str] | None = None,
    ) -> tuple[ModelSpec, dict[str, Any]]:
        avoid = avoid or set()
        eligible, excluded = self.eligible_models(
            profile, available_providers
        )

        # Record models skipped because a previous attempt failed
        # or the caller explicitly asked us to avoid them.
        remaining = []
        for model in eligible:
            key = f"{model.provider}:{model.model_id}"
            if key in avoid:
                excluded.append({
                    "model": model.model_id,
                    "provider": model.provider,
                    "reason": "Skipped because this model was already attempted or explicitly avoided",
                })
            else:
                remaining.append(model)

        eligible = remaining

        if not eligible:
            self.trace.append({
                "event": "routing_blocked",
                "task_type": profile.task_type,
                "excluded": excluded,
            })
            raise RoutingBlocked(
                "No eligible model. Check task requirements, "
                "provider availability, and inference policy."
            )

        # Apply the actual routing policy.
        paid_cost_estimates = {}
        if self.mode == "live" and self.allow_paid:
            for model in eligible:
                paid_cost_estimates[model.provider + ":" + model.model_id] = (
                    model.estimate_cost(1000, 500)
                )

            eligible.sort(
                key=lambda m: (
                    paid_cost_estimates[
                        m.provider + ":" + m.model_id
                    ]
                    if paid_cost_estimates[
                        m.provider + ":" + m.model_id
                    ] is not None
                    else float("inf"),
                    -m.quality_score,
                    m.latency_score,
                )
            )
            selection_method = (
                "Lowest verified estimated cost first; quality and "
                "latency break ties."
            )
        else:
            eligible.sort(
                key=lambda m: (
                    0 if m.tier == "economy" else 1,
                    -m.quality_score,
                    m.latency_score,
                )
            )
            selection_method = (
                "Prefer economy-tier candidates, then higher configured "
                "quality scores, then lower configured latency scores."
            )

        chosen = eligible[0]
        chosen_key = f"{chosen.provider}:{chosen.model_id}"

        candidate_comparison = []
        for rank, model in enumerate(eligible, start=1):
            key = f"{model.provider}:{model.model_id}"

            if rank == 1:
                ranking_reason = (
                    "Ranked first by the active routing policy."
                )
            elif self.mode == "live" and self.allow_paid:
                ranking_reason = (
                    "Ranked below the selected model by the estimated-cost "
                    "policy or its quality/latency tie-breakers."
                )
            else:
                ranking_reason = (
                    "Ranked below the selected model by tier preference, "
                    "configured quality score, or configured latency score."
                )

            candidate_comparison.append({
                "rank": rank,
                "provider": model.provider,
                "model": model.model_id,
                "tier": model.tier,
                "quality_score": model.quality_score,
                "quality_required": profile.quality_required,
                "latency_score": model.latency_score,
                "max_latency_score": profile.max_latency_score,
                "capability_match": (
                    profile.task_type == "general"
                    or profile.task_type in model.capabilities
                ),
                "pricing_known": model.pricing_known,
                "estimated_cost_per_1000_input_500_output_usd": (
                    paid_cost_estimates.get(key)
                    if self.mode == "live" and self.allow_paid
                    else None
                ),
                "free_tier_eligible_in_local_config": (
                    model.free_tier_verified
                ),
                "metadata_source": model.metadata_source,
                "decision": (
                    "SELECTED" if key == chosen_key else "Eligible alternative"
                ),
                "ranking_reason": ranking_reason,
            })

        if self.mode == "live" and self.allow_paid:
            selection_reason = (
                "Passed eligibility checks and ranked first by estimated "
                "cost, with quality and latency as tie-breakers."
            )
        else:
            selection_reason = (
                "Passed eligibility checks and ranked first under the "
                "economy-tier, quality-score, and latency-score policy."
            )

        rationale = {
            "provider": chosen.provider,
            "model": chosen.model_id,
            "tier": chosen.tier,
            "reason": selection_reason,
            "selection_method": (
                "runtime-discovered catalog + task policy"
                if self.mode == "live"
                else "synthetic mock catalog + task policy"
                ),
            "selection_policy_explanation": selection_method,
            "task_profile": asdict(profile),
            "selection_factors": {
                "task_type": profile.task_type,
                "complexity": profile.complexity,
                "required_quality": profile.quality_required,
                "sensitivity": profile.sensitivity,
                "max_latency_score": profile.max_latency_score,
                "required_capability": profile.task_type,
                "paid_inference_enabled": self.allow_paid,
                "verified_free_inference_enabled": self.allow_verified_free,
            },
            "candidate_comparison": candidate_comparison,
            "excluded_models": excluded,
            "pricing_known": chosen.pricing_known,
            "free_tier_eligible_in_local_config": (
                chosen.free_tier_verified
            ),
            "metadata_source": chosen.metadata_source,
            "mode": self.mode,
            "cost_note": (
                "Candidate cost is an estimate for 1,000 input and 500 output "
                "tokens, not the actual cost of this request."
                if self.mode == "live" and self.allow_paid
                else
                "No price-based ranking was performed. Unknown pricing "
                "does not mean the model is free."
            ),
            "metadata_note": (
                "Quality, tier, capabilities, and free eligibility may come "
                "from local configuration; they are not independently "
                "verified by the selection algorithm."
            ),
        }

        self.trace.append({
            "event": "model_selected",
            **rationale,
        })
        return chosen, rationale

    def generate(self, profile: TaskProfile, prompt: str, provider_clients: dict[str, Any], avoid: set[str] | None = None) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            raise RoutingBlocked(f"Maximum model-call limit ({self.max_calls}) reached")
        if self.session_spend >= self.session_budget and self.mode == "live" and self.allow_paid:
            raise RoutingBlocked("Session budget exhausted")
        attempted = set(avoid or set())
        last_error = None
        while self.calls < self.max_calls:
            model, rationale = self.select(profile, attempted, set(provider_clients))
            # Conservative preflight budget check before any live paid request. Token estimates are
            # deliberately padded; unknown prices were already excluded during eligibility filtering.
            if self.mode == "live" and self.allow_paid:
                estimated_input_tokens = max(1, len(prompt) // 3)
                estimated_output_tokens = 1000
                preflight_cost = model.estimate_cost(estimated_input_tokens, estimated_output_tokens)
                if preflight_cost is None:
                    raise RoutingBlocked("Cannot call model because its pricing is unknown")
                if preflight_cost > self.max_task_cost:
                    raise RoutingBlocked(f"Preflight estimate ${preflight_cost:.6f} exceeds per-task limit ${self.max_task_cost:.6f}")
                if self.session_spend + preflight_cost > self.session_budget:
                    raise RoutingBlocked(f"Preflight estimate ${preflight_cost:.6f} exceeds remaining session budget")
            self.calls += 1
            try:
                result = provider_clients[model.provider].generate(model.model_id, prompt)
                break
            except Exception as exc:
                last_error = exc
                attempted.add(f"{model.provider}:{model.model_id}")
                self.trace.append({"event": "provider_failure", "provider": model.provider, "model": model.model_id,
                                   "error_type": type(exc).__name__, "message": "Provider call failed; trying another eligible model if available."})
        else:
            raise RoutingBlocked(f"All eligible provider attempts failed or call limit was reached ({type(last_error).__name__ if last_error else 'no candidate'})")
        usage = result.get("usage", {})
        cost = model.estimate_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        if self.mode == "live" and self.allow_paid:
            if cost is None:
                # Never pretend an unknown price is zero. Block unpriced model before calls in production;
                # this is a defensive reporting guard if provider usage is incomplete.
                self.trace.append({"event": "cost_unknown", "model": model.model_id})
            elif cost > self.max_task_cost or self.session_spend + cost > self.session_budget:
                self.trace.append({"event": "budget_overrun_detected", "model": model.model_id, "estimated_cost_usd": cost})
            else:
                self.session_spend += cost
        record = {"model": model.model_id, "provider": model.provider, "text": result["text"], "usage": usage,
                  "latency_seconds": result.get("latency_seconds"), "estimated_cost_usd": cost,
                  "cost_status": "estimated" if cost is not None else "unknown", "rationale": rationale}
        self.trace.append({"event": "model_call_completed", **{k: v for k, v in record.items() if k != "text"}})
        return record
