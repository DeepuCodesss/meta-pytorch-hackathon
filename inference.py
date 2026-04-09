#!/usr/bin/env python3
"""
PhantomShield X – Inference Script (Scaler x OpenEnv Hackathon compliant)

Required environment variables:
    API_BASE_URL      The OpenAI-compatible LLM endpoint base URL
    MODEL_NAME        The model identifier to use for inference
    HF_TOKEN          Your HuggingFace / API key (no default — must be set)

Optional environment variables:
    LOCAL_IMAGE_NAME  Docker image name if using from_docker_image()
    TASK              easy | medium | hard | all  (default: all)
    VERBOSE           1 for verbose stderr output

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama3-8b-8192"
    export HF_TOKEN="gsk_..."
    python inference.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from environment.env import PhantomShieldEnv, SystemState
from tasks.tasks import TaskRunner, TASK_REGISTRY
from graders.graders import get_grader, GradeResult

# ---------------------------------------------------------------------------
# Environment variable setup (with required defaults per checklist)
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "llama3-8b-8192")
HF_TOKEN         = os.getenv("HF_TOKEN")           # NO default — must be set explicitly
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # Optional: used with from_docker_image()

# ---------------------------------------------------------------------------
# Structured stdout logging — mandatory START / STEP / END format
# ---------------------------------------------------------------------------

def log_start(task_id: str, task_name: str, model: str, api_base_url: str):
    # Required structured block — OpenEnv Phase 2 validator scans stdout for [START]
    print(f"[START] task={task_name} task_id={task_id} model={model}", flush=True)
    # Full JSON record for tooling
    record = {
        "event":        "START",
        "task_id":      task_id,
        "task_name":    task_name,
        "model":        model,
        "api_base_url": api_base_url,
    }
    print(json.dumps(record), flush=True)


def log_step(task_id: str, step: int, action: str, target: Any,
             reward: float, cumulative_reward: float, reasoning: str):
    # Required structured block — OpenEnv Phase 2 validator scans stdout for [STEP]
    print(
        f"[STEP] task_id={task_id} step={step} action={action} "
        f"target={target} reward={round(reward, 4)} "
        f"cumulative_reward={round(cumulative_reward, 4)}",
        flush=True,
    )
    # Full JSON record for tooling
    record = {
        "event":             "STEP",
        "task_id":           task_id,
        "step":              step,
        "action":            action,
        "target":            target,
        "reward":            reward,
        "cumulative_reward": round(cumulative_reward, 4),
        "reasoning":         reasoning,
    }
    print(json.dumps(record), flush=True)


def log_end(task_id: str, steps: int, cumulative_reward: float,
            score: float, label: str, breakdown: Dict, feedback: str):
    # Required structured block — OpenEnv Phase 2 validator scans stdout for [END]
    print(
        f"[END] task_id={task_id} steps={steps} score={round(score, 4)} "
        f"label={label} cumulative_reward={round(cumulative_reward, 4)}",
        flush=True,
    )
    # Full JSON record for tooling
    record = {
        "event":             "END",
        "task_id":           task_id,
        "steps":             steps,
        "cumulative_reward": round(cumulative_reward, 4),
        "score":             score,
        "label":             label,
        "breakdown":         breakdown,
        "feedback":          feedback,
    }
    print(json.dumps(record), flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a cybersecurity analyst AI. You will be given a JSON snapshot of a
network monitoring system containing:
  - login_events: recent login attempts
  - file_events: file access activity
  - network_events: network traffic
  - alerts: triggered security alerts
  - sessions: active user sessions
  - flagged_ips / blocked_ips: already-actioned IPs
  - step: current step number

Your job is to choose ONE action per turn. Respond ONLY with a valid JSON object:
{
  "action": "<action>",
  "target": "<ip_or_alert_id_or_null>",
  "reasoning": "<one-line explanation>"
}

Valid actions:
  ignore           - No action needed
  flag_suspicious  - Mark an IP as suspicious (requires target IP)
  block_ip         - Block an IP address (requires target IP)
  escalate         - Escalate an alert to security team (requires target alert_id)

Rules:
- Repeated failed logins from same IP -> flag_suspicious or block_ip
- Off-hours login from unusual location -> flag_suspicious then escalate
- Multiple correlated low-severity alerts from same IP -> escalate then block
- Set target to null only for 'ignore'
- Output ONLY the JSON object, nothing else
""".strip()


# ---------------------------------------------------------------------------
# LLM Agent — uses OpenAI client with API_BASE_URL + MODEL_NAME + HF_TOKEN
# ---------------------------------------------------------------------------

class LLMAgent:
    """LLM agent using the OpenAI client configured via environment variables."""

    def __init__(self):
        if not HF_TOKEN:
            raise EnvironmentError(
                "HF_TOKEN is not set. Export your API key before running."
            )
        self.model = MODEL_NAME
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )

    def decide(self, state: SystemState) -> Dict[str, Any]:
        state_json = state.model_dump()
        state_json.pop("threat_detected", None)
        state_json.pop("done", None)

        user_msg = (
            f"Current system state:\n{json.dumps(state_json, indent=2)}"
            f"\n\nWhat is your action?"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())


# ---------------------------------------------------------------------------
# Heuristic fallback agent — deterministic, no API key required
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """Rule-based fallback agent. Runs automatically when HF_TOKEN is unset."""

    def decide(self, state: SystemState) -> Dict[str, Any]:
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        # HARD: 2+ alerts from same IP -> escalate each in turn, then block
        ip_alert_map: Dict[str, list] = {}
        for a in state.alerts:
            if a.source_ip:
                ip_alert_map.setdefault(a.source_ip, []).append(a)

        for ip, alerts in ip_alert_map.items():
            if len(alerts) >= 2:
                pending = [a for a in alerts if a.alert_id not in state.escalated_alerts]
                if pending:
                    top = max(pending, key=lambda a: severity_order.get(a.severity, 0))
                    return {
                        "action":    "escalate",
                        "target":    top.alert_id,
                        "reasoning": f"APT chain: {len(alerts)} alerts from {ip}, escalating {top.alert_id}.",
                    }
                if ip not in state.blocked_ips:
                    return {
                        "action":    "block_ip",
                        "target":    ip,
                        "reasoning": f"All alerts from {ip} escalated — blocking.",
                    }

        # EASY: block IP with most failed logins
        ip_fail: Dict[str, int] = {}
        for ev in state.login_events:
            if not ev.success:
                ip_fail[ev.ip] = ip_fail.get(ev.ip, 0) + 1
        if ip_fail:
            worst = max(ip_fail, key=lambda ip: ip_fail[ip])
            if ip_fail[worst] >= 3 and worst not in state.blocked_ips:
                return {
                    "action":    "block_ip",
                    "target":    worst,
                    "reasoning": f"{worst} has {ip_fail[worst]} failed logins.",
                }

        # MEDIUM: escalate high/critical alerts; flag then escalate medium
        pending_alerts = [a for a in state.alerts if a.alert_id not in state.escalated_alerts]
        if pending_alerts:
            top = max(pending_alerts, key=lambda a: severity_order.get(a.severity, 0))
            if top.severity in ("critical", "high"):
                return {
                    "action":    "escalate",
                    "target":    top.alert_id,
                    "reasoning": f"Escalating {top.severity} alert {top.alert_id}.",
                }
            if top.severity == "medium" and top.source_ip:
                if top.source_ip not in state.flagged_ips:
                    return {
                        "action":    "flag_suspicious",
                        "target":    top.source_ip,
                        "reasoning": "Flagging source IP of medium-severity alert.",
                    }
                return {
                    "action":    "escalate",
                    "target":    top.alert_id,
                    "reasoning": f"IP already flagged — escalating {top.alert_id}.",
                }

        # Catch unusual login locations
        for ev in state.login_events:
            if ev.location not in ("HQ", "HQ-VPN", "Unknown") and ev.success:
                if ev.ip not in state.flagged_ips and ev.ip not in state.blocked_ips:
                    return {
                        "action":    "flag_suspicious",
                        "target":    ev.ip,
                        "reasoning": f"Unusual login location: {ev.location}.",
                    }

        return {"action": "ignore", "target": None, "reasoning": "No threat detected."}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, agent, verbose: bool = False) -> Dict[str, Any]:
    runner   = TaskRunner(task_id)
    grader   = get_grader(task_id)
    task_spec = TASK_REGISTRY[task_id]

    state            = runner.reset()
    action_log: List[Dict] = []
    cumulative_reward = 0.0
    step             = 0

    log_start(
        task_id      = task_id,
        task_name    = task_spec.name,
        model        = MODEL_NAME,
        api_base_url = API_BASE_URL,
    )

    if verbose:
        print(runner.describe(), file=sys.stderr)

    while True:
        try:
            decision = agent.decide(state)
        except Exception as e:
            decision = {"action": "ignore", "target": None, "reasoning": f"Agent error: {e}"}

        action    = decision.get("action",    "ignore")
        target    = decision.get("target")
        reasoning = decision.get("reasoning", "")

        result             = runner.step(action=action, target=target)
        cumulative_reward += result.reward
        step              += 1

        action_log.append({
            "step":   step,
            "action": action,
            "target": target,
            "reward": result.reward,
            "info":   result.info,
        })

        log_step(
            task_id           = task_id,
            step              = step,
            action            = action,
            target            = target,
            reward            = result.reward,
            cumulative_reward = cumulative_reward,
            reasoning         = reasoning,
        )

        if verbose:
            print(
                f"  Step {step:2d} | {action:16s} | target={str(target):20s}"
                f" | reward={result.reward:+.1f} | {reasoning}",
                file=sys.stderr,
            )

        state = result.state
        if result.done:
            break

    grade: GradeResult = grader.grade(
        final_state       = state,
        cumulative_reward = cumulative_reward,
        steps_taken       = step,
        action_log        = action_log,
    )

    log_end(
        task_id           = task_id,
        steps             = step,
        cumulative_reward = cumulative_reward,
        score             = grade.score,
        label             = grade.label,
        breakdown         = grade.breakdown,
        feedback          = grade.feedback,
    )

    return {
        "task_id":           task_id,
        "steps":             step,
        "cumulative_reward": round(cumulative_reward, 4),
        "score":             grade.score,
        "label":             grade.label,
        "breakdown":         grade.breakdown,
        "feedback":          grade.feedback,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    verbose  = os.getenv("VERBOSE", "0").strip() == "1"
    task_arg = os.getenv("TASK",    "all").strip().lower()

    # Choose agent
    if HF_TOKEN:
        print(f"[INFO] LLM agent | model={MODEL_NAME} | base={API_BASE_URL}", file=sys.stderr)
        try:
            agent = LLMAgent()
        except Exception as e:
            print(f"[WARN] LLM init failed ({e}) — using heuristic fallback.", file=sys.stderr)
            agent = HeuristicAgent()
    else:
        print("[INFO] HF_TOKEN not set — using heuristic baseline agent.", file=sys.stderr)
        agent = HeuristicAgent()

    tasks_to_run = ["easy", "medium", "hard"] if task_arg == "all" else [task_arg]

    results = []
    for tid in tasks_to_run:
        print(f"\n[TASK] {tid.upper()}", file=sys.stderr)
        result = run_episode(tid, agent, verbose=verbose)
        results.append(result)
        print(f"  Score : {result['score']:.4f}  [{result['label']}]", file=sys.stderr)
        print(f"  Steps : {result['steps']}", file=sys.stderr)
        print(f"  Reward: {result['cumulative_reward']}", file=sys.stderr)

    if len(results) > 1:
        avg = round(sum(r["score"] for r in results) / len(results), 4)
        print(f"\n  Average Score: {avg:.4f}", file=sys.stderr)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
