import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from agents.coordinator import CoordinatorAgent
from providers.mock_provider import MockProvider
from router.engine import Router, RoutingBlocked
from router.models import build_registry, discover_registry

PROVIDER_ADAPTERS = [
    ("openai", "providers.openai_provider", "OpenAIProvider", "OPENAI_API_KEY"),
    ("groq", "providers.groq_provider", "GroqProvider", "GROQ_API_KEY"),
    ("anthropic", "providers.anthropic_provider", "AnthropicProvider", "ANTHROPIC_API_KEY"),
    ("gemini", "providers.gemini_provider", "GeminiProvider", "GEMINI_API_KEY or GOOGLE_API_KEY"),
]

def make_providers(mode: str):
    if mode == "mock":
        return {name: MockProvider(name) for name in ("openai", "groq", "anthropic", "gemini")}
    clients, errors = {}, []
    for name, module, cls, env_name in PROVIDER_ADAPTERS:
        if name == "gemini":
            has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        else:
            has_key = bool(os.getenv(env_name))
        if not has_key:
            continue
        try:
            imported = __import__(module, fromlist=[cls])
            clients[name] = getattr(imported, cls)()
        except Exception as exc:
            # Do not echo arbitrary provider exception text; it could contain request details.
            errors.append(f"{name}: {type(exc).__name__}")
    if not clients:
        raise RuntimeError("No live provider could be initialized. Add one or more provider keys to the local .env file. " + "; ".join(errors))
    if errors:
        print("Provider initialization warnings:", "; ".join(errors), file=sys.stderr)
    return clients

def print_registry(registry, mode):
    print("MODEL CATALOG MODE:", "SIMULATED — IDs are examples, not real provider models" if mode == "mock" else "LIVE DISCOVERY — IDs returned by configured provider APIs")
    print(json.dumps([{"provider": m.provider, "model_id": m.model_id, "routing_tier": m.tier,
                       "heuristic_quality_score": m.quality_score, "capabilities": list(m.capabilities),
                       "metadata_source": m.metadata_source, "pricing_known": m.pricing_known,
                       "free_tier_verified": m.free_tier_verified} for m in registry], indent=2))

def main():
    parser = argparse.ArgumentParser(description="Dynamic Multi-Agent LLM Router prototype")
    parser.add_argument("--task", help="Task to run")
    parser.add_argument("--sensitivity", choices=["public", "internal", "sensitive", "restricted"], default="public")
    parser.add_argument("--mode", choices=["mock", "live"], default=os.getenv("ROUTER_MODE", "mock"))
    parser.add_argument("--show-models", action="store_true", help="Show synthetic mock catalog or discover live model IDs; no inference calls")
    args = parser.parse_args()
    try:
        if args.mode == "mock":
            providers = make_providers("mock")
            registry = build_registry()
        else:
            providers = make_providers("live")
            registry = discover_registry(providers)
        if args.show_models:
            print_registry(registry, args.mode)
            return
        task = args.task or input("Enter a task for the agents: ").strip()
        if not task:
            print("No task supplied.", file=sys.stderr); sys.exit(2)
        allow_paid = os.getenv("ALLOW_PAID_INFERENCE", "false").lower() == "true"
        allow_free = os.getenv("ALLOW_VERIFIED_FREE_TIER_INFERENCE", "false").lower() == "true"
        if args.mode == "live" and not allow_paid and not allow_free:
            print("LIVE INFERENCE BLOCKED: free eligibility is not verified and paid calls are disabled. Model discovery does not establish pricing or free-tier access.")
            sys.exit(3)
        router = Router(registry, mode=args.mode, allow_paid=allow_paid, allow_verified_free=allow_free,
                        max_calls=int(os.getenv("MAX_CALLS_PER_TASK", "5")),
                        max_task_cost=float(os.getenv("MAX_TASK_COST_USD", "0")),
                        session_budget=float(os.getenv("SESSION_BUDGET_USD", "0")))
        result = CoordinatorAgent(router, providers).run(task, args.sensitivity)
        print("\n=== PROTOTYPE RESULT ===")
        print("MODE:", "SIMULATED (not a real LLM response)" if args.mode == "mock" else "LIVE PROVIDER")
        print("TASK PROFILE:", json.dumps(result["task_profile"], indent=2))
        print("\nFINAL ANSWER:\n", result["final_answer"])
        print("\nVERIFICATION:\n", json.dumps({k:v for k,v in result["verification"].items() if k != "model_review"}, indent=2))
        print("\nROUTING TRACE:\n", json.dumps(result["trace"], indent=2))
        print("\nSESSION SPEND USD:", result["session_spend_usd"])
    except RoutingBlocked as exc:
        print("ROUTING BLOCKED:", exc); sys.exit(4)
    except Exception as exc:
        print(f"Execution failed: {type(exc).__name__}: {exc}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
