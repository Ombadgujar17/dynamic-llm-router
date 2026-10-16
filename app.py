
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from agents.coordinator import CoordinatorAgent
from main import make_providers
from router.engine import Router, RoutingBlocked
from router.models import build_registry, discover_registry


st.set_page_config(
    page_title="Dynamic LLM Router",
    page_icon="🔀",
    layout="wide",
)

st.title("🔀 Dynamic Multi-Agent LLM Router")
st.caption(
    "Task-aware model selection, multi-agent coordination, "
    "and cost-conscious routing."
)

with st.sidebar:
    st.header("Configuration")
    mode_label = st.radio(
        "Execution mode",
        ["Mock (safe demo)", "Live (provider API)"],
    )
    mode = "mock" if mode_label.startswith("Mock") else "live"

    sensitivity = st.selectbox(
        "Data sensitivity",
        ["public", "internal", "sensitive", "restricted"],
    )

    st.info(
        "Mock mode uses simulated models and responses. "
        "Live mode requires configured provider credentials "
        "and eligible model metadata."
    )

task = st.text_area(
    "What should the agents do?",
    placeholder="Example: Classify this customer message and explain why.",
    height=120,
)

run_col, discovery_col = st.columns(2)
run_clicked = run_col.button(
    "▶ Run task", type="primary", use_container_width=True
)
discover_clicked = discovery_col.button(
    "🔎 Discover models", use_container_width=True
)


def get_runtime(selected_mode):
    providers = make_providers(selected_mode)
    registry = (
        build_registry()
        if selected_mode == "mock"
        else discover_registry(providers)
    )
    return providers, registry


def show_model_decision(stage, call_result):
    st.markdown(f"### {stage}")

    if not call_result:
        st.info("No model call recorded for this stage.")
        return

    rationale = call_result.get("rationale", {})
    usage = call_result.get("usage", {})

    left, middle, right = st.columns(3)
    left.metric("Provider", rationale.get("provider", call_result.get("provider", "Unknown")))
    middle.metric("Model", rationale.get("model", call_result.get("model", "Unknown")))
    right.metric("Tier", rationale.get("tier", "Unknown"))

    st.markdown("**Why this model was selected**")
    st.write(rationale.get("reason", "No selection rationale recorded."))

    factors = rationale.get("selection_factors", {})
    if factors:
        st.markdown("**Task requirements used by the router**")
        st.json(factors)

    st.caption(
        rationale.get(
            "selection_policy_explanation",
            rationale.get("selection_method", "Selection policy unavailable."),
            )
        )
    st.caption(rationale.get("cost_note", "Cost-ranking details unavailable."))
    st.caption(rationale.get("metadata_note", ""))

    candidates = rationale.get("candidate_comparison", [])
    if candidates:
        with st.expander(
            f"Eligible candidates ({len(candidates)})",
            expanded=True,
        ):
            st.dataframe(
                candidates,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning(
            "No candidate comparison was recorded for this selection."
        )

    excluded = rationale.get("excluded_models", [])
    if excluded:
        with st.expander(f"Why {len(excluded)} candidate(s) were excluded or skipped"):
            st.dataframe(
                excluded,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("**Execution details**")
    st.write({
        "Simulated": usage.get("simulated", "Not reported"),
        "Latency (seconds)": call_result.get("latency_seconds"),
        "Estimated cost (USD)": call_result.get("estimated_cost_usd"),
        "Cost status": call_result.get("cost_status", "Unknown"),
    })


if discover_clicked:
    try:
        providers, registry = get_runtime(mode)
        st.subheader("Model catalog")
        st.caption(
            "Mock entries are synthetic. Live entries are discovered "
            "from configured provider APIs; discovery does not prove "
            "capabilities, pricing, or free-tier eligibility."
        )
        st.dataframe(
            [
                {
                    "Provider": model.provider,
                    "Model ID": model.model_id,
                    "Tier": model.tier,
                    "Quality estimate": model.quality_score,
                    "Capabilities": ", ".join(model.capabilities),
                    "Pricing known": model.pricing_known,
                    "Free tier verified": model.free_tier_verified,
                    "Metadata source": model.metadata_source,
                }
                for model in registry
            ],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error(f"Model discovery failed ({type(exc).__name__}).")
        st.caption(
            "Check your provider configuration and local environment."
        )


if run_clicked:
    if not task.strip():
        st.warning("Enter a task before running the agents.")
    else:
        try:
            providers, registry = get_runtime(mode)

            allow_paid = (
                os.getenv("ALLOW_PAID_INFERENCE", "false").lower()
                == "true"
            )
            allow_free = (
                os.getenv(
                    "ALLOW_VERIFIED_FREE_TIER_INFERENCE", "false"
                ).lower()
                == "true"
            )

            if mode == "live" and not allow_paid and not allow_free:
                st.error(
                    "Live inference is blocked. Paid calls are disabled, "
                    "and verified free-tier inference has not been enabled."
                )
                st.stop()

            router = Router(
                registry,
                mode=mode,
                allow_paid=allow_paid,
                allow_verified_free=allow_free,
                max_calls=int(os.getenv("MAX_CALLS_PER_TASK", "5")),
                max_task_cost=float(
                    os.getenv("MAX_TASK_COST_USD", "0")
                ),
                session_budget=float(
                    os.getenv("SESSION_BUDGET_USD", "0")
                ),
            )

            with st.spinner("Running the agent workflow..."):
                result = CoordinatorAgent(
                    router, providers
                ).run(task.strip(), sensitivity)

            st.session_state["last_result"] = result
            st.session_state["last_mode"] = mode

        except RoutingBlocked as exc:
            st.error(f"Routing blocked: {exc}")
        except Exception as exc:
            st.error(f"Task failed ({type(exc).__name__}).")
            st.caption(
                "Check provider configuration and application logs."
            )


result = st.session_state.get("last_result")

if result:
    result_mode = st.session_state.get("last_mode", "mock")

    st.divider()
    st.subheader("Execution results")

    if result_mode == "mock":
        st.warning("SIMULATED — this is not a real LLM response.")
    else:
        st.success("Live provider execution completed.")

    selected_events = [
        event for event in result["trace"]
        if event.get("event") == "model_selected"
    ]

    first, second, third = st.columns(3)
    first.metric("Model calls", len([
        event for event in result["trace"]
        if event.get("event") == "model_call_completed"
    ]))
    second.metric(
        "Session spend",
        f"${result['session_spend_usd']:.6f}",
    )
    third.metric(
        "Models selected",
        len({
            (event.get("provider"), event.get("model"))
            for event in selected_events
        }),
    )

    st.subheader("Final answer")
    st.write(result["final_answer"])

    with st.expander("Agent workflow"):
        st.markdown("**Execution plan**")
        st.write(result["plan"]["text"])

        st.markdown("**Analysis draft**")
        st.write(result["analysis"]["text"])

        st.markdown("**Verification result**")
        verification = {
            key: value
            for key, value in result["verification"].items()
            if key != "model_review"
        }
        st.json(verification)


    with st.expander("Why each model was selected", expanded=True):
        show_model_decision("1. Planning Agent", result.get("plan"))
        st.divider()

        show_model_decision("2. Analysis Agent", result.get("analysis"))
        st.divider()

        verification = result.get("verification", {})
        show_model_decision(
            "3. Verification Agent",
            verification.get("model_review"),
        )

        escalation = result.get("escalation")
        if escalation:
            st.divider()
            show_model_decision("4. Escalation / Revision", escalation)


st.divider()
st.caption(
    "Prototype disclaimer: model quality scores are estimates. "
    "Verification checks do not guarantee factual correctness."
)
