# 🔀 Dynamic Multi-Agent LLM Router

A dynamic LLM routing engine that decouples agents from hardcoded models. At runtime, it analyzes task complexity, capabilities, and data sensitivity to route each agentic step (Planning, Analysis, Verification) to the optimal model across **OpenAI, Groq, Anthropic Claude, and Google Gemini**.

---

## ⚡ Key Highlights

- **Dynamic Cross-Provider Routing**: Routes across OpenAI, Groq, Anthropic, and Gemini without fixed model pinning.
- **Multi-Agent Pipeline**: Planning (economy tier) → Analysis (specialized capability) → Verification (evaluator) → Revision/Escalation (on failure).
- **Dynamic Model Discovery**: Fetches available models directly from provider APIs and filters out non-text models.
- **Safety & Cost Guardrails**: Live paid calls blocked by default, preflight budget checks, and automatic cross-provider fallback on failure.
- **Dual Interfaces**: Interactive Streamlit Web UI and scriptable CLI.

---

## 🚀 Quickstart & Setup

### 1. Installation

```bash
# Clone and enter repo
git clone https://github.com/Ombadgujar17/dynamic-llm-router.git
cd dynamic-llm-router

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### 2. Configure Environment (`.env`)

Add keys for any providers you want to use (optional for mock mode):

```dotenv
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Mode: 'mock' (simulated, no keys/cost) or 'live' (real provider APIs)
ROUTER_MODE=mock

# Live inference controls (disabled by default for safety)
ALLOW_PAID_INFERENCE=false
ALLOW_VERIFIED_FREE_TIER_INFERENCE=false
```

---

## 🖥 How to Run

### Option A: Streamlit Web Dashboard (Recommended)

Run the interactive web interface:

```bash
streamlit run app.py
```

- Toggle between **Mock** and **Live** modes.
- Click **🔎 Discover models** to see available models and capabilities.
- Enter any task to inspect the multi-agent execution and the **"Why each model was selected"** decision breakdown.

### Option B: Command-Line Interface (CLI)

#### 1. Safe Mock Mode (No API keys or internet needed)

```bash
# General task
python main.py --mode mock --task "Classify this customer message as positive or negative"

# Coding task
python main.py --mode mock --task "Write a Python function to validate an email address"

# Sensitive data task (restricts provider dispatch)
python main.py --mode mock --task "Analyze patient lab records" --sensitivity sensitive

# View available mock models
python main.py --mode mock --show-models
```

#### 2. Live Mode (Using Real Provider APIs)

```bash
# Discover live model IDs visible to your configured API keys (Listing only, no text generation)
python main.py --mode live --show-models

# Run live inference with verified free models
# (Requires VERIFIED_FREE_MODEL_IDS and ALLOW_VERIFIED_FREE_TIER_INFERENCE=true in .env)
python main.py --mode live --task "Summarize quantum computing in 2 sentences"
```

---

## 🛠 Core Configuration Reference

| Environment Variable | Default | Purpose |
|---|---|---|
| `ROUTER_MODE` | `mock` | `mock` (simulated demo) or `live` (provider APIs) |
| `ALLOW_PAID_INFERENCE` | `false` | Must be `true` to execute live paid model calls |
| `ALLOW_VERIFIED_FREE_TIER_INFERENCE` | `false` | Must be `true` to execute verified free-tier models |
| `MAX_CALLS_PER_TASK` | `5` | Hard limit on model invocations per task |
| `SENSITIVE_DATA_ALLOWED_PROVIDERS` | `openai` | Allowed providers when sensitivity is `sensitive` or `restricted` |
| `MAX_TASK_COST_USD` | `0.00` | Max spend cap for a single task |
| `SESSION_BUDGET_USD` | `0.00` | Cumulative spend cap for the active session |
| `VERIFIED_FREE_MODEL_IDS` | *empty* | Comma-separated list of free models (e.g. `gemini:gemini-1.5-flash`) |

---

## 🧪 Running Tests

Run the test suite (100% offline, simulated provider clients, no keys required):

```bash
python -m unittest discover -s tests -v
```

---

## 📂 Project Structure

```
dynamic-llm-router/
├── app.py                # Streamlit Web UI
├── main.py               # CLI runner
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
├── router/
│   ├── engine.py         # Filtering, ranking, budget checks & failover
│   ├── models.py         # Model discovery & ModelSpec schema
│   └── policies.py       # Task classifier & TaskProfile schema
├── agents/
│   ├── coordinator.py    # Master pipeline orchestrator
│   ├── analysis.py       # Core analytical worker
│   └── verification.py   # Output auditor & validation agent
├── providers/            # Adapters (OpenAI, Groq, Anthropic, Gemini, Mock)
└── tests/
    └── test_router.py    # Automated test cases
```
