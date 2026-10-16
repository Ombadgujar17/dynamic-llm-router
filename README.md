# Dynamic Multi-Agent LLM Router — Multi-provider prototype

A small Python CLI prototype with a Coordinator, Analysis Agent, Verification Agent, and one central runtime router. It supports **OpenAI, Groq, Anthropic Claude, and Google Gemini** through provider adapters. Add a key for any provider to `.env`; live discovery uses the provider's model-list API and routes requests across eligible discovered models. There is no fixed model-per-agent mapping.

## How routing works

1. Classify the task and estimate complexity, required quality, sensitivity, and latency needs.
2. Discover models exposed to configured API credentials.
3. Filter candidates by provider availability, task capabilities, sensitivity allowlist, latency, quality threshold, and budget rules.
4. Select among eligible candidates, preferring the inferred economy tier for simple tasks and stronger candidates for demanding tasks. If verified price metadata is configured and paid calls are enabled, compare estimated cost first.
5. Execute through the chosen provider adapter, verify the response, and fall back/escalate within a bounded call limit if required.
6. Record model choice, rationale, excluded candidates, usage, latency, and estimated cost in the trace.

## Important limitations (state these honestly in an interview)

- `mock` mode uses a **synthetic catalog and simulated responses**. Its model IDs are examples, not real provider models.
- `live` mode discovers IDs returned by configured provider APIs. Listing a model does not prove it is free, supports every task, or has a particular quality level.
- Model tier/quality metadata inferred from IDs is a **heuristic**, not a benchmark. Cross-provider quality should ultimately be measured with a shared evaluation set.
- Prices and free-tier eligibility are unknown by default. Live inference is blocked unless verified free eligibility is explicitly configured or paid inference is enabled after reviewing billing.
- Sensitive/restricted tasks default to an OpenAI-only allowlist as a conservative example. Review data processing terms and your own policy before allowing another provider using `SENSITIVE_DATA_ALLOWED_PROVIDERS`.
- Token usage and latency are provider-reported/measured where available. Cost is an estimate based on configured prices, not the provider's final invoice.
- Verification is a lightweight prototype check, not a guarantee of factual correctness.

## Setup on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add only the keys for providers you want to use. Never paste keys into chat, commit `.env`, or print keys in logs. `.gitignore` excludes `.env`.

| Provider | Environment variable | Official model discovery docs |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | https://developers.openai.com/api/reference/resources/models/methods/list |
| Groq | `GROQ_API_KEY` | https://console.groq.com/docs/models |
| Anthropic Claude | `ANTHROPIC_API_KEY` | https://platform.claude.com/docs/en/api/go/models/list |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | https://ai.google.dev/gemini-api/docs/models |

## Safe mock demo (no keys, no inference calls)

```bash
python main.py --mode mock --task "Classify this customer message as positive, neutral, or negative"
python main.py --mode mock --task "Write a Python function to validate an email address"
python main.py --mode mock --show-models
```

Mock output and token counts are simulated. The mock catalog includes examples for all four providers to demonstrate cross-provider routing; it is not a claim that these exact models exist.

## Discover live model IDs (model listing only)

1. Create keys from the official provider dashboards.
2. Put them locally in `.env` under the matching environment variables.
3. Run:

```bash
python main.py --mode live --show-models
```

This calls model-list endpoints for providers whose keys are configured. It does not intentionally call text-generation endpoints. Some provider/model-list APIs can differ in what they expose; treat returned IDs as discovered availability, not proof of inference permissions or free usage.

## Enable inference

Live inference is blocked by default. To use a free-tier model, independently verify the provider's current terms and your account eligibility, add its exact discovered model ID to `VERIFIED_FREE_MODEL_IDS` using `provider:model-id` format, and set `ALLOW_VERIFIED_FREE_TIER_INFERENCE=true`. Do this only when you have confirmed the allowance.

If you intentionally permit paid inference, set `ALLOW_PAID_INFERENCE=true`, a non-zero per-task/session budget, and verified pricing in `MODEL_METADATA_JSON`. Do not use these settings unless you accept potential charges. Unknown prices remain blocked.

Example model metadata format (use only facts you have verified):

```dotenv
MODEL_METADATA_JSON={"gemini:EXACT_MODEL_ID":{"quality_score":4,"tier":"economy","capabilities":["general","classification","summarization","coding","reasoning","verification"],"input_usd_per_million":0.1,"output_usd_per_million":0.4,"free_tier_verified":false}}
```

Do not copy the example prices as actual pricing; replace them with verified current values or leave pricing unknown.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use fake provider clients and simulated outputs; no live keys or paid calls are required.

## Architecture

- `router/models.py`: live model discovery, cautious ID heuristics, optional metadata overrides, mock catalog.
- `router/policies.py`: explainable task profiling.
- `router/engine.py`: eligibility, runtime selection, budget checks, bounded retries/fallback, routing trace.
- `providers/`: OpenAI, Groq, Anthropic, Gemini, and mock adapters.
- `agents/`: Coordinator, Analysis, Verification.
- `main.py`: CLI runner.

## Interview explanation

The router does not pin a model to an agent. It discovers the models available through configured provider APIs, filters them using task requirements, privacy rules, quality threshold, latency and budget, then chooses a candidate at runtime. It can select across OpenAI, Groq, Anthropic and Gemini. Current cross-provider quality/tier estimates are explicitly heuristic; a production router would populate capabilities and prices from verified metadata and use an evaluation set to calibrate model quality. Free eligibility is a separate account/provider constraint and is never inferred from a model name.
