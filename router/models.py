"""Runtime model discovery and conservative metadata inference.

Provider model IDs are discovered at runtime in live mode. The mock catalog is
synthetic and exists only to demonstrate routing without API calls.
"""
from dataclasses import dataclass
from typing import Optional, Any
import json
import os
import re

TASK_CAPABILITIES = ("general", "classification", "summarization", "coding", "reasoning", "verification")

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    tier: str  # inferred economy/strong/unknown; not an official provider label
    capabilities: tuple[str, ...]
    quality_score: int  # heuristic estimate, not a benchmark
    latency_score: int  # heuristic estimate; lower is faster
    input_usd_per_million: Optional[float] = None
    output_usd_per_million: Optional[float] = None
    free_tier_verified: bool = False
    metadata_source: str = "heuristic"

    @property
    def pricing_known(self) -> bool:
        return self.input_usd_per_million is not None and self.output_usd_per_million is not None

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        if not self.pricing_known:
            return None
        return (input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million) / 1_000_000



def _model_heuristics(model_id: str) -> tuple[str, int, int, tuple[str, ...]]:
    """
    Model discovery does not prove model capabilities or quality.
    Unknown models remain ineligible until their metadata is configured.

    Returns:
        tier, quality_score, latency_score, capabilities

    A quality score of 0 means unknown, not poor quality.
    """
    name = model_id.lower()

    # Recognizable specialized models that should not enter the
    # general-purpose text-generation registry.
    non_text_markers = (
        "embedding",
        "embed-",
        "whisper",
        "tts",
        "text-to-speech",
        "transcribe",
        "moderation",
        "prompt-guard",
        "orpheus",
        "dall-e",
        "image-generation",
    )

    if any(marker in name for marker in non_text_markers):
        return "unsupported", 0, 5, ()

    # A model ID alone cannot reliably establish its capabilities,
    # quality, price, or speed. Require explicit metadata instead.
    return "unknown", 0, 5, ()



def _metadata_overrides() -> dict[str, dict[str, Any]]:
    raw = os.getenv("MODEL_METADATA_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def discover_registry(providers: dict[str, Any]) -> list[ModelSpec]:
    """Use each configured provider's model-list endpoint; never invent live IDs."""
    overrides = _metadata_overrides()
    verified_free = {item.strip() for item in os.getenv("VERIFIED_FREE_MODEL_IDS", "").split(",") if item.strip()}
    registry: list[ModelSpec] = []
    errors: list[str] = []
    for provider_name, client in providers.items():
        try:
            model_ids = client.list_models()
        except Exception as exc:
            errors.append(f"{provider_name}: {type(exc).__name__}")
            continue
        for model_id in sorted(set(model_ids)):
            tier, quality, latency, capabilities = _model_heuristics(model_id)
            if tier == "unsupported":
                continue
            override = overrides.get(f"{provider_name}:{model_id}", {})
            if not isinstance(override, dict):
                override = {}
            try:
                quality = int(override.get("quality_score", quality))
                latency = int(override.get("latency_score", latency))
            except (TypeError, ValueError):
                pass
            capabilities = tuple(override.get("capabilities", capabilities))
            tier = str(override.get("tier", tier))
            in_price = override.get("input_usd_per_million")
            out_price = override.get("output_usd_per_million")
            try:
                in_price = float(in_price) if in_price is not None else None
                out_price = float(out_price) if out_price is not None else None
            except (TypeError, ValueError):
                in_price, out_price = None, None
            free_verified = bool(override.get("free_tier_verified", False)) or f"{provider_name}:{model_id}" in verified_free or model_id in verified_free
            registry.append(
                ModelSpec(
                    provider=provider_name,
                    model_id=model_id,
                    tier=tier,
                    capabilities=capabilities,
                    quality_score=max(0, min(5, quality)),
                    latency_score=max(1, min(5, latency)),
                    input_usd_per_million=in_price,
                    output_usd_per_million=out_price,
                    free_tier_verified=free_verified,
                    metadata_source=(
                        "explicit-config"
                        if override
                        else "unverified-discovery"
                    ),
                )
            )
    if not registry:
        detail = "; ".join(errors) if errors else "providers returned no compatible text-generation models"
        raise RuntimeError("Could not discover any usable text-generation models. " + detail)
    return registry


def build_registry() -> list[ModelSpec]:
    """Synthetic models for mock mode only; not claimed to exist at either provider."""
    return [
        ModelSpec("groq", "mock-economy-8b", "economy", TASK_CAPABILITIES, 3, 1, metadata_source="simulated-catalog"),
        ModelSpec("openai", "mock-economy-mini", "economy", TASK_CAPABILITIES, 3, 2, metadata_source="simulated-catalog"),
        ModelSpec("gemini", "mock-economy-flash", "economy", TASK_CAPABILITIES, 3, 1, metadata_source="simulated-catalog"),
        ModelSpec("anthropic", "mock-economy-haiku", "economy", TASK_CAPABILITIES, 3, 2, metadata_source="simulated-catalog"),
        ModelSpec("groq", "mock-strong-70b", "strong", TASK_CAPABILITIES, 4, 3, metadata_source="simulated-catalog"),
        ModelSpec("openai", "mock-strong-reasoning", "strong", TASK_CAPABILITIES, 5, 4, metadata_source="simulated-catalog"),
        ModelSpec("gemini", "mock-strong-pro", "strong", TASK_CAPABILITIES, 5, 3, metadata_source="simulated-catalog"),
        ModelSpec("anthropic", "mock-strong-sonnet", "strong", TASK_CAPABILITIES, 5, 3, metadata_source="simulated-catalog"),
    ]
