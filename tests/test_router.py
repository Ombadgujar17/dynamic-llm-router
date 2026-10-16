
import json
import os
import unittest
from unittest.mock import patch

from router.engine import Router, RoutingBlocked
from router.models import ModelSpec, discover_registry
from router.policies import classify_task
from providers.mock_provider import MockProvider


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.registry = [
            ModelSpec(
                "groq", "cheap", "economy",
                ("general", "classification"), 3, 1
            ),
            ModelSpec(
                "openai", "strong", "strong",
                ("general", "reasoning"), 5, 3
            ),
        ]

    def test_live_discovery_uses_provider_returned_ids(self):
        class CatalogProvider:
            def __init__(self, ids):
                self.ids = ids

            def list_models(self):
                return self.ids

        registry = discover_registry({
            "openai": CatalogProvider([
                "provider-returned-mini",
                "provider-returned-reasoning",
            ]),
            "groq": CatalogProvider([
                "provider-returned-8b",
                "provider-returned-70b",
            ]),
        })

        ids = {model.model_id for model in registry}

        self.assertEqual(ids, {
            "provider-returned-mini",
            "provider-returned-reasoning",
            "provider-returned-8b",
            "provider-returned-70b",
        })
        self.assertNotIn("gpt-4.1-mini", ids)

        for model in registry:
            self.assertEqual(
                model.metadata_source, "unverified-discovery"
            )
            self.assertEqual(model.quality_score, 0)
            self.assertEqual(model.capabilities, ())

    def test_discovered_catalog_routes_simple_and_complex_differently(self):
        class CatalogProvider:
            def __init__(self, ids):
                self.ids = ids

            def list_models(self):
                return self.ids

        metadata = {
            "groq:vendor-small-8b": {
                "tier": "economy",
                "quality_score": 3,
                "latency_score": 1,
                "capabilities": ["general", "classification"],
            },
            "groq:vendor-large-70b": {
                "tier": "strong",
                "quality_score": 4,
                "latency_score": 3,
                "capabilities": ["general", "coding", "reasoning"],
            },
            "openai:vendor-mini": {
                "tier": "economy",
                "quality_score": 3,
                "latency_score": 2,
                "capabilities": ["general", "classification"],
            },
            "openai:vendor-pro-reasoning": {
                "tier": "strong",
                "quality_score": 5,
                "latency_score": 3,
                "capabilities": ["general", "coding", "reasoning"],
            },
        }

        with patch.dict(os.environ, {
            "MODEL_METADATA_JSON": json.dumps(metadata)
        }):
            registry = discover_registry({
                "groq": CatalogProvider([
                    "vendor-small-8b",
                    "vendor-large-70b",
                ]),
                "openai": CatalogProvider([
                    "vendor-mini",
                    "vendor-pro-reasoning",
                ]),
            })

        router = Router(registry, mode="mock")

        simple, _ = router.select(
            classify_task("Classify this short note")
        )
        complex_model, _ = router.select(
            classify_task(
                "Write a Python function to solve a complex coding problem"
            )
        )

        self.assertEqual(simple.tier, "economy")
        self.assertEqual(complex_model.tier, "strong")
        self.assertIn(
            "coding", complex_model.capabilities
        )

    def test_mock_catalog_covers_four_provider_families(self):
        from router.models import build_registry

        providers = {
            model.provider for model in build_registry()
        }
        self.assertEqual(
            providers, {"openai", "groq", "anthropic", "gemini"}
        )

    def test_router_can_select_across_provider_families(self):
        registry = [
            ModelSpec("openai", "small", "economy",
                      ("classification",), 3, 2),
            ModelSpec("anthropic", "haiku", "economy",
                      ("classification",), 3, 1),
            ModelSpec("gemini", "flash", "economy",
                      ("classification",), 3, 1),
            ModelSpec("groq", "8b", "economy",
                      ("classification",), 3, 1),
        ]

        router = Router(registry, mode="mock")
        selected, trace = router.select(
            classify_task("Classify this message")
        )

        self.assertIn(
            selected.provider,
            {"openai", "groq", "anthropic", "gemini"}
        )
        self.assertEqual(
            trace["selection_method"],
            "synthetic mock catalog + task policy"
        )

    def test_simple_task_prefers_economy(self):
        router = Router(self.registry, mode="mock")
        model, _ = router.select(
            classify_task("Classify this message")
        )
        self.assertEqual(model.model_id, "cheap")

    def test_complex_task_requires_higher_quality(self):
        profile = classify_task(
            "Explain a complex medical diagnosis and patient treatment"
        )
        router = Router(self.registry, mode="mock")
        eligible, _ = router.eligible_models(profile)

        self.assertTrue(
            all(m.quality_score >= profile.quality_required
                for m in eligible)
        )
        self.assertTrue(
            any(m.model_id == "strong" for m in eligible)
        )

    def test_live_mode_blocks_unverified_free_model(self):
        router = Router(
            self.registry,
            mode="live",
            allow_paid=False,
            allow_verified_free=False,
        )
        with self.assertRaises(RoutingBlocked):
            router.select(
                classify_task("Summarize this paragraph")
            )

    def test_sensitive_task_excludes_groq(self):
        router = Router(self.registry, mode="mock")
        profile = classify_task(
            "Summarize this record", sensitivity="sensitive"
        )
        eligible, excluded = router.eligible_models(profile)

        self.assertTrue(
            all(m.provider == "openai" for m in eligible)
        )
        self.assertTrue(
            any(x["provider"] == "groq" for x in excluded)
        )

    def test_mock_call_is_marked_simulated(self):
        router = Router(self.registry, mode="mock")
        profile = classify_task(
            "Classify this short note"
        )
        provider = {
            "groq": MockProvider("groq"),
            "openai": MockProvider("openai"),
        }

        result = router.generate(
            profile, "Classify: hello world", provider
        )
        self.assertTrue(result["usage"]["simulated"])

    def test_capability_filter_excludes_incompatible_model(self):
        registry = [
            ModelSpec(
                "groq", "classification-only", "economy",
                ("classification",), 4, 1
            ),
            ModelSpec(
                "openai", "reasoning", "strong",
                ("reasoning",), 5, 3
            ),
        ]

        router = Router(registry, mode="mock")
        profile = classify_task("Summarize this text")
        eligible, excluded = router.eligible_models(profile)

        self.assertEqual(eligible, [])
        self.assertEqual(len(excluded), 2)

    def test_provider_failure_falls_back_to_another_model(self):
        class FailingProvider:
            def generate(self, model, prompt):
                raise TimeoutError("simulated timeout")

        registry = [
            ModelSpec(
                "openai", "first", "economy",
                ("classification",), 4, 1
            ),
            ModelSpec(
                "groq", "second", "economy",
                ("classification",), 3, 2
            ),
        ]

        router = Router(
            registry, mode="mock", max_calls=3
        )
        profile = classify_task("Classify this message")
        providers = {
            "openai": FailingProvider(),
            "groq": MockProvider("groq"),
        }

        result = router.generate(
            profile, "classify this", providers
        )

        self.assertEqual(result["provider"], "groq")
        self.assertTrue(
            any(row["event"] == "provider_failure"
                for row in router.trace)
        )


def test_verification_intent_takes_priority_over_coding_keywords(self):
    profile = classify_task(
        "Verify whether this Python function is correct"
    )
    self.assertEqual(profile.task_type, "verification")
    self.assertEqual(profile.quality_required, 4)


if __name__ == "__main__":
    unittest.main()
