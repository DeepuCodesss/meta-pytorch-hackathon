# 🛡 PhantomShield X
### AI Cyber Defense Training Environment — Scaler × OpenEnv Hackathon

An OpenEnv-compliant simulation where an AI agent acts as a cybersecurity analyst,
detecting and responding to network threats by analysing logs, alerts, and traffic.

---

## Environment Variables (Mandatory for LLM inference)

| Variable | Description |
|---|---|
| `API_BASE_URL` | Base URL of the OpenAI-compatible LLM endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `llama3-8b-8192`) |
| `HF_TOKEN` | HuggingFace / API key |
| `TASK` | Task to run: `easy` \| `medium` \| `hard` \| `all` (default: `all`) |
| `VERBOSE` | Set to `1` for verbose per-step output to stderr |

> If `API_BASE_URL`, `MODEL_NAME`, or `HF_TOKEN` are not set, the script automatically
> falls back to the built-in **heuristic baseline agent** (no API key needed).

---

## Project Structure

```
phantomshield-x/
├── environment/env.py      # OpenEnv core: reset(), step(), state()
├── tasks/tasks.py          # 3 tasks (easy / medium / hard) + TaskRunner
├── graders/graders.py      # Deterministic graders returning scores in [0.0, 1.0]
├── inference.py            # LLM agent (OpenAI client) + heuristic baseline
├── app.py                  # Gradio UI (HuggingFace Spaces entry point)
├── openenv.yaml            # OpenEnv spec (metadata, tasks, endpoints, rewards)
├── Dockerfile              # Container build
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama3-8b-8192"
export HF_TOKEN="gsk_..."
```

---

## Running Inference

### With LLM agent (env vars must be set)

```bash
python inference.py
# or specific task:
TASK=easy python inference.py
TASK=hard VERBOSE=1 python inference.py
```

### Heuristic fallback (no API key needed)

```bash
python inference.py   # auto-detects missing env vars and uses heuristic
```

### Gradio web UI

```bash
python app.py
# Open http://localhost:7860
```

---

## Structured Stdout Log Format

`inference.py` emits one JSON object per line to stdout for each event:

```jsonc
// Episode start
{"event": "START", "task_id": "easy", "task_name": "Brute Force Login Detection", "model": "llama3-8b-8192", "api_base_url": "..."}

// Each step
{"event": "STEP", "task_id": "easy", "step": 1, "action": "block_ip", "target": "192.168.1.105", "reward": 1.0, "cumulative_reward": 1.0, "reasoning": "IP has 8 failed logins."}

// Episode end
{"event": "END", "task_id": "easy", "steps": 1, "cumulative_reward": 1.0, "score": 1.0, "label": "Excellent", "breakdown": {...}, "feedback": "..."}
```

---

## Tasks

| ID | Name | Difficulty | Threat IP |
|---|---|---|---|
| easy | Brute Force Login Detection | Easy | 192.168.1.105 |
| medium | After-Hours Geo-Anomaly | Medium | 203.0.113.77 |
| hard | Multi-Stage APT Attack | Hard | 198.51.100.42 |

### Easy
8 failed logins from the same IP → attacker eventually succeeds. Flag or block the IP.

### Medium
User 'charlie' logs in at 3 AM from Eastern Europe and immediately reads sensitive HR files.
Detect the compromised account — escalate `ALERT-002` or block the IP.

### Hard
User 'eve' executes a kill chain: VPN login → sudoers read → cron backdoor → privilege escalation → 52 MB exfiltration.
Correlate 3+ low/medium alerts from the same IP. ALERT-006 (privilege escalation) appears at step 3.

---

## OpenEnv Interface

```python
from environment.env import PhantomShieldEnv

env = PhantomShieldEnv(task_id="easy")   # easy | medium | hard
state = env.reset()                      # returns SystemState
result = env.step("block_ip", target="192.168.1.105")
# result.state, result.reward, result.done, result.info
current = env.state()
```

### Actions

| Action | Target | Description |
|---|---|---|
| ignore | None | No action |
| flag_suspicious | IP address | Mark IP for monitoring |
| block_ip | IP address | Block all traffic from IP |
| escalate | alert_id | Escalate alert to security team |

### Reward

| Event | Reward |
|---|---|
| Correct detection (≤3 steps) | +1.0 |
| Correct detection (late) | +0.5 |
| False positive | -0.5 |
| Missed threat (timeout) | -1.0 |
| Neutral action | 0.0 |

---

## Docker

```bash
# Build
docker build -t phantomshield-x .

# Run with LLM agent
docker run \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama3-8b-8192" \
  -e HF_TOKEN="gsk_..." \
  phantomshield-x

# Run Gradio UI
docker run -p 7860:7860 \
  -e API_BASE_URL="..." -e MODEL_NAME="..." -e HF_TOKEN="..." \
  phantomshield-x python app.py

# Run heuristic only (no API key)
docker run phantomshield-x
```

---

## HuggingFace Spaces Deployment

Push to HuggingFace Hub — the Gradio UI starts automatically via `app.py`.

Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` as **Space Secrets** to enable
the LLM agent. Without them, the heuristic baseline runs automatically.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

_PhantomShield X — Built for Scaler × OpenEnv Hackathon_
