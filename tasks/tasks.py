"""
PhantomShield X – Task Definitions
Each task wraps the environment with task-specific metadata and success criteria.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment.env import PhantomShieldEnv, SystemState


@dataclass
class TaskSpec:
    task_id: str
    name: str
    difficulty: str
    description: str
    goal: str
    max_steps: int
    threat_ip: str
    threat_user: Optional[str]
    key_indicators: List[str]
    expected_actions: List[str]
    hints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "easy": TaskSpec(
        task_id="easy",
        name="Brute Force Login Detection",
        difficulty="Easy",
        description=(
            "An external IP is hammering the admin account with repeated failed logins. "
            "Eventually, the attacker succeeds. Identify and respond to the threat."
        ),
        goal="Flag or block the attacker IP before or shortly after successful login.",
        max_steps=10,
        threat_ip="192.168.1.105",
        threat_user="admin",
        key_indicators=[
            "8 consecutive failed logins from 192.168.1.105",
            "Login attempts within a 16-minute window",
            "Alert ALERT-001: high severity brute force warning",
        ],
        expected_actions=["flag_suspicious", "block_ip"],
        hints=[
            "Look for repeated failed logins from the same IP.",
            "Check ALERT-001 — it is a high-severity warning.",
            "The threat IP is 192.168.1.105",
        ],
    ),

    "medium": TaskSpec(
        task_id="medium",
        name="After-Hours Geo-Anomaly Detection",
        difficulty="Medium",
        description=(
            "User 'charlie' normally logs in from HQ during business hours. "
            "A login occurred at 3:14 AM from Eastern Europe, followed by access to sensitive HR files. "
            "Determine whether this is legitimate and respond appropriately."
        ),
        goal="Detect the compromised account activity and escalate or block.",
        max_steps=10,
        threat_ip="203.0.113.77",
        threat_user="charlie",
        key_indicators=[
            "Login at 03:14 AM vs normal 9–5 window",
            "Login location: Eastern Europe (never seen before)",
            "Immediate access to /data/payroll and /data/hr files",
            "Alert ALERT-002: medium severity geo-anomaly",
        ],
        expected_actions=["flag_suspicious", "block_ip", "escalate"],
        hints=[
            "Compare the login time with charlie's normal activity pattern.",
            "Check the login location — it has never appeared before.",
            "Sensitive files were accessed within 2 minutes of login.",
        ],
    ),

    "hard": TaskSpec(
        task_id="hard",
        name="Multi-Stage APT Attack",
        difficulty="Hard",
        description=(
            "User 'eve' logged in via VPN and performed a series of individually "
            "low-severity actions that together form a classic APT attack chain: "
            "reconnaissance → privilege escalation → data exfiltration. "
            "Correlate the alerts and respond before data leaves the network."
        ),
        goal="Correlate multi-step attack chain and escalate before full exfiltration.",
        max_steps=10,
        threat_ip="198.51.100.42",
        threat_user="eve",
        key_indicators=[
            "Read /etc/sudoers (ALERT-003 — low severity)",
            "Modified /etc/cron.d (ALERT-004 — low severity)",
            "52 MB outbound transfer to 91.108.4.200 (ALERT-005 — medium)",
            "Privilege escalation attempt revealed at step 3 (ALERT-006 — high)",
        ],
        expected_actions=["escalate", "block_ip"],
        hints=[
            "No single alert is alarming alone — look at the sequence.",
            "Three low/medium alerts from the same source IP spell an APT.",
            "At step 3 a critical escalation alert appears — act on it.",
            "The threat IP is 198.51.100.42",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

class TaskRunner:
    """Convenience wrapper — creates env + task spec together."""

    def __init__(self, task_id: str):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Options: {list(TASK_REGISTRY)}")
        self.spec = TASK_REGISTRY[task_id]
        self.env = PhantomShieldEnv(task_id=task_id)
        self.env._max_steps = self.spec.max_steps

    def reset(self) -> SystemState:
        return self.env.reset()

    def step(self, action: str, target: Optional[str] = None):
        return self.env.step(action, target)

    def state(self) -> SystemState:
        return self.env.state()

    def describe(self) -> str:
        s = self.spec
        return (
            f"\n{'='*60}\n"
            f"Task    : {s.name}\n"
            f"Difficulty: {s.difficulty}\n"
            f"Goal    : {s.goal}\n"
            f"Max Steps: {s.max_steps}\n"
            f"{'='*60}\n"
            f"{s.description}\n"
        )


if __name__ == "__main__":
    for tid in ("easy", "medium", "hard"):
        runner = TaskRunner(tid)
        print(runner.describe())
