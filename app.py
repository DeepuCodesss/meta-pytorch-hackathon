"""
PhantomShield X – Entry Point
Serves BOTH:
  • FastAPI REST API  (OpenEnv checker endpoints)  — /reset, /state, /step
  • Gradio UI                                       — /ui/
Both live on the same port (7860) via a single uvicorn server.
"""

import json
import os
import sys
import threading

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from environment.env import PhantomShieldEnv
from tasks.tasks import TaskRunner, TASK_REGISTRY
from graders.graders import get_grader
from inference import HeuristicAgent, LLMAgent, run_episode


# ---------------------------------------------------------------------------
# FastAPI app + OpenEnv REST endpoints
# ---------------------------------------------------------------------------

api = FastAPI(title="PhantomShield X OpenEnv API", version="1.0.0")

# Global environment store — the checker is effectively single-session,
# so we keep one env per task_id and always return its state.
_env_store: dict[str, PhantomShieldEnv] = {}
_runner_store: dict[str, TaskRunner] = {}


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy", description="Task difficulty: easy | medium | hard")


class StepRequest(BaseModel):
    action: str = Field(..., description="One of: ignore, flag_suspicious, block_ip, escalate")
    target: Optional[str] = Field(default=None, description="IP address or alert_id")


def _get_runner(task_id: str) -> TaskRunner:
    if task_id not in _runner_store:
        raise HTTPException(status_code=400, detail=f"No active session for task '{task_id}'. Call /reset first.")
    return _runner_store[task_id]


@api.get("/")
def root():
    return {"status": "ok", "service": "PhantomShield X OpenEnv API", "version": "1.0.0"}


@api.post("/reset")
def reset(body: ResetRequest = None):
    """Initialise (or restart) the environment. Returns the initial SystemState."""
    task_id = (body.task_id if body else None) or "easy"
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: easy, medium, hard")
    runner = TaskRunner(task_id)
    _runner_store[task_id] = runner
    state = runner.reset()
    return JSONResponse(content=state.model_dump(mode="json"))


@api.get("/state")
def get_state(task_id: str = "easy"):
    """Return the current SystemState without advancing the environment."""
    runner = _get_runner(task_id)
    state = runner.state()
    return JSONResponse(content=state.model_dump(mode="json"))


@api.post("/step")
def step(body: StepRequest, task_id: str = "easy"):
    """Execute one agent action and return StepResult."""
    runner = _get_runner(task_id)
    try:
        result = runner.step(action=body.action, target=body.target or None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    payload = {
        "state": result.state.model_dump(mode="json"),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Session state helpers (Gradio)
# ---------------------------------------------------------------------------

def make_fresh_session():
    return {
        "runner": None,
        "action_log": [],
        "cumulative_reward": 0.0,
        "step": 0,
        "done": False,
        "task_id": "easy",
    }


def format_state(state) -> str:
    if state is None:
        return "No active session. Click **Reset** to start."
    return json.dumps(state.model_dump(), indent=2, default=str)


def start_task(task_id: str, session: dict):
    session = make_fresh_session()
    session["task_id"] = task_id
    runner = TaskRunner(task_id)
    session["runner"] = runner
    state = runner.reset()
    task_desc = TASK_REGISTRY[task_id]
    info_md = (
        f"### 🛡 Task: {task_desc.name}  `{task_desc.difficulty}`\n\n"
        f"{task_desc.description}\n\n"
        f"**Goal:** {task_desc.goal}\n\n"
        f"**Hints:**\n" + "\n".join(f"- {h}" for h in task_desc.hints)
    )
    return (
        session,
        format_state(state),
        info_md,
        "Task started. Analyse the state and choose an action.",
        "",
    )


def take_action(action: str, target: str, session: dict):
    if session.get("runner") is None:
        return session, "⚠ Please reset / start a task first.", "", ""

    if session["done"]:
        return session, "Episode is finished. Reset to play again.", "", ""

    runner: TaskRunner = session["runner"]
    target = target.strip() if target else None
    if not target:
        target = None

    try:
        result = runner.step(action=action, target=target)
    except ValueError as e:
        return session, f"Error: {e}", "", ""

    session["cumulative_reward"] += result.reward
    session["step"] += 1
    log_entry = {
        "step": session["step"],
        "action": action,
        "target": target,
        "reward": result.reward,
        "info": result.info,
    }
    session["action_log"].append(log_entry)

    status_msg = (
        f"Step {session['step']} | Action: `{action}` | Target: `{target}` | "
        f"Reward: **{result.reward:+.1f}** | Cumulative: **{session['cumulative_reward']:+.1f}**"
    )
    if result.state.message:
        status_msg += f"\n\n{result.state.message}"

    grade_md = ""
    if result.done:
        session["done"] = True
        grader = get_grader(session["task_id"])
        grade = grader.grade(
            final_state=result.state,
            cumulative_reward=session["cumulative_reward"],
            steps_taken=session["step"],
            action_log=session["action_log"],
        )
        grade_md = (
            f"## 🏁 Episode Complete\n\n"
            f"**Score:** `{grade.score:.4f}` — **{grade.label}**\n\n"
            f"**Breakdown:**\n"
            + "\n".join(f"- {k}: {v:.1f}" for k, v in grade.breakdown.items())
            + f"\n\n**Feedback:**\n```\n{grade.feedback}\n```"
        )

    return session, format_state(result.state), status_msg, grade_md


def run_heuristic_demo(task_id: str, session: dict):
    """Run the baseline agent on a fresh episode (LLM if env vars set, else heuristic)."""
    api_base = os.environ.get("API_BASE_URL", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    if api_base and hf_token and model_name:
        try:
            agent = LLMAgent()
        except Exception:
            agent = HeuristicAgent()
    else:
        agent = HeuristicAgent()
    result = run_episode(task_id, agent, verbose=False)
    out = (
        f"## Heuristic Agent Results — {task_id.upper()}\n\n"
        f"**Score:** `{result['score']:.4f}` [{result['label']}]\n\n"
        f"**Steps:** {result['steps']}\n\n"
        f"**Cumulative Reward:** {result['cumulative_reward']}\n\n"
        f"**Breakdown:**\n"
        + "\n".join(f"- {k}: {v:.1f}" for k, v in result["breakdown"].items())
        + f"\n\n**Feedback:**\n```\n{result['feedback']}\n```"
    )
    return out


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#title { text-align: center; }
#state_box { font-family: monospace; font-size: 0.78rem; }
"""

with gr.Blocks(css=CSS, title="PhantomShield X") as demo:
    session_state = gr.State(make_fresh_session())

    gr.Markdown(
        "# 🛡 PhantomShield X\n"
        "### AI Cyber Defense Training Environment\n"
        "_Detect and respond to cybersecurity threats. Choose a task, analyse the system state, and take actions._",
        elem_id="title",
    )

    with gr.Row():
        task_selector = gr.Radio(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Select Task Difficulty",
        )
        reset_btn = gr.Button("🔄 Reset / Start Task", variant="primary")

    task_info_md = gr.Markdown("Select a task and click **Reset** to begin.")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📋 System State (JSON)")
            state_display = gr.Code(
                label="",
                language="json",
                elem_id="state_box",
                lines=28,
            )
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Take Action")
            action_selector = gr.Radio(
                choices=["ignore", "flag_suspicious", "block_ip", "escalate"],
                value="ignore",
                label="Action",
            )
            target_input = gr.Textbox(
                label="Target (IP address or alert_id — leave blank for 'ignore')",
                placeholder="e.g. 192.168.1.105 or ALERT-001",
            )
            action_btn = gr.Button("▶ Execute Action", variant="secondary")

            status_display = gr.Markdown("_Status will appear here._")
            grade_display = gr.Markdown("")

    with gr.Accordion("🤖 Run Heuristic Baseline Agent", open=False):
        gr.Markdown("Runs the rule-based baseline agent automatically on any task.")
        heuristic_task = gr.Radio(
            choices=["easy", "medium", "hard"], value="easy", label="Task"
        )
        run_heuristic_btn = gr.Button("Run Heuristic Agent")
        heuristic_output = gr.Markdown("")

    # ── Event bindings ──────────────────────────────────────────────────────
    reset_btn.click(
        fn=start_task,
        inputs=[task_selector, session_state],
        outputs=[session_state, state_display, task_info_md, status_display, grade_display],
    )

    action_btn.click(
        fn=take_action,
        inputs=[action_selector, target_input, session_state],
        outputs=[session_state, state_display, status_display, grade_display],
    )

    run_heuristic_btn.click(
        fn=run_heuristic_demo,
        inputs=[heuristic_task, session_state],
        outputs=[heuristic_output],
    )

    gr.Markdown(
        "_PhantomShield X — Built for Scaler OpenEnv Hackathon · "
        "[GitHub](https://github.com/DeepuCodesss/meta-pytorch-hackathon)_"
    )


# ---------------------------------------------------------------------------
# Mount Gradio onto FastAPI and launch with uvicorn
# ---------------------------------------------------------------------------

# Mount the Gradio app at /ui so the root path is free for the REST API
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
