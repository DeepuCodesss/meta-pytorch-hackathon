"""
PhantomShield X – Core OpenEnv Environment
Simulates a cybersecurity monitoring system with log-based threat detection.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Typed Models
# ---------------------------------------------------------------------------

class LoginEvent(BaseModel):
    timestamp: str
    user: str
    ip: str
    success: bool
    location: str
    attempts: int = 1

class FileAccessEvent(BaseModel):
    timestamp: str
    user: str
    file_path: str
    action: str  # read / write / delete / copy
    sensitive: bool = False

class NetworkEvent(BaseModel):
    timestamp: str
    src_ip: str
    dst_ip: str
    port: int
    bytes_transferred: int
    protocol: str

class Alert(BaseModel):
    alert_id: str
    severity: str          # low / medium / high / critical
    message: str
    source_ip: Optional[str] = None
    user: Optional[str] = None
    timestamp: str

class SessionData(BaseModel):
    session_id: str
    user: str
    ip: str
    login_time: str
    active: bool
    privilege_level: str   # normal / elevated / admin

class SystemState(BaseModel):
    step: int
    task_id: str
    login_events: List[LoginEvent] = Field(default_factory=list)
    file_events: List[FileAccessEvent] = Field(default_factory=list)
    network_events: List[NetworkEvent] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)
    sessions: List[SessionData] = Field(default_factory=list)
    flagged_ips: List[str] = Field(default_factory=list)
    blocked_ips: List[str] = Field(default_factory=list)
    escalated_alerts: List[str] = Field(default_factory=list)
    threat_detected: bool = False
    done: bool = False
    message: str = ""

class StepResult(BaseModel):
    state: SystemState
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_CORRECT_DETECTION   =  1.0
REWARD_LATE_DETECTION      =  0.5
REWARD_FALSE_POSITIVE      = -0.5
REWARD_MISSED_THREAT       = -1.0
REWARD_NEUTRAL             =  0.0

# ---------------------------------------------------------------------------
# PhantomShield Environment
# ---------------------------------------------------------------------------

class PhantomShieldEnv:
    """OpenEnv-compliant cybersecurity training environment."""

    VALID_ACTIONS = {"ignore", "flag_suspicious", "block_ip", "escalate"}

    def __init__(self, task_id: str = "easy"):
        assert task_id in ("easy", "medium", "hard"), \
            f"task_id must be one of: easy, medium, hard. Got: {task_id}"
        self.task_id = task_id
        self._state: Optional[SystemState] = None
        self._threat_ip: Optional[str] = None
        self._threat_user: Optional[str] = None
        self._attack_stage: int = 0          # used for hard multi-step task
        self._max_steps: int = 10
        self._step_count: int = 0
        self._early_detection_threshold: int = 3  # steps before "late"
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> SystemState:
        """Reset the environment and return initial state."""
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._attack_stage = 0

        if self.task_id == "easy":
            self._state = self._build_easy_scenario()
        elif self.task_id == "medium":
            self._state = self._build_medium_scenario()
        else:
            self._state = self._build_hard_scenario()

        return self._state

    def step(self, action: str, target: Optional[str] = None) -> StepResult:
        """
        Execute an action and return (new_state, reward, done, info).

        action  : one of VALID_ACTIONS
        target  : IP address or alert_id the action targets (optional)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Choose from {self.VALID_ACTIONS}")

        self._step_count += 1
        reward, info = self._apply_action(action, target)
        self._cumulative_reward += reward

        # Advance scenario
        self._advance_scenario()

        # Terminal conditions
        # Task-specific completion criteria
        if self.task_id == "hard":
            # Hard: must escalate at least one alert AND block the IP
            fully_resolved = (
                len(self._state.escalated_alerts) > 0
                and self._threat_ip in self._state.blocked_ips
            )
        elif self.task_id == "medium":
            # Medium: escalate ALERT-002 OR block the threat IP
            fully_resolved = (
                "ALERT-002" in self._state.escalated_alerts
                or self._threat_ip in self._state.blocked_ips
            )
        else:
            fully_resolved = self._state.threat_detected

        done = fully_resolved or self._step_count >= self._max_steps

        if done:
            if not (self._state.threat_detected or fully_resolved):
                # Missed the threat entirely
                reward += REWARD_MISSED_THREAT
                self._cumulative_reward += REWARD_MISSED_THREAT
                info["missed_threat"] = True
            self._state.done = True

        self._state.step = self._step_count
        return StepResult(
            state=self._state,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> SystemState:
        """Return current environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    def _ts(self, offset_minutes: int = 0) -> str:
        base = datetime(2024, 6, 15, 10, 0, 0)
        return (base + timedelta(minutes=offset_minutes)).isoformat()

    def _build_easy_scenario(self) -> SystemState:
        """
        Easy: Obvious brute-force — 8 failed logins from same IP.
        """
        attacker_ip = "192.168.1.105"
        self._threat_ip = attacker_ip

        logins = [
            LoginEvent(
                timestamp=self._ts(i * 2),
                user="admin",
                ip=attacker_ip,
                success=False,
                location="Unknown",
                attempts=i + 1,
            )
            for i in range(8)
        ]
        # One eventual success
        logins.append(LoginEvent(
            timestamp=self._ts(18),
            user="admin",
            ip=attacker_ip,
            success=True,
            location="Unknown",
            attempts=9,
        ))

        # Normal background traffic
        normal = [
            LoginEvent(timestamp=self._ts(-5), user="alice", ip="10.0.0.2",
                       success=True, location="HQ", attempts=1),
            LoginEvent(timestamp=self._ts(-10), user="bob", ip="10.0.0.5",
                       success=True, location="HQ", attempts=1),
        ]

        alert = Alert(
            alert_id="ALERT-001",
            severity="high",
            message="Multiple failed login attempts detected for user 'admin'",
            source_ip=attacker_ip,
            user="admin",
            timestamp=self._ts(16),
        )

        return SystemState(
            step=0,
            task_id="easy",
            login_events=normal + logins,
            alerts=[alert],
            sessions=[
                SessionData(session_id="S001", user="alice", ip="10.0.0.2",
                            login_time=self._ts(-5), active=True, privilege_level="normal"),
            ],
        )

    def _build_medium_scenario(self) -> SystemState:
        """
        Medium: Login from unusual location at 3 AM + access to sensitive files.
        """
        attacker_ip = "203.0.113.77"
        attacker_user = "charlie"
        self._threat_ip = attacker_ip
        self._threat_user = attacker_user

        logins = [
            # Normal daytime logins
            LoginEvent(timestamp=self._ts(-120), user="charlie", ip="10.0.0.8",
                       success=True, location="HQ", attempts=1),
            LoginEvent(timestamp=self._ts(-200), user="diana", ip="10.0.0.9",
                       success=True, location="HQ", attempts=1),
            # Suspicious: 3 AM login from foreign IP
            LoginEvent(
                timestamp="2024-06-15T03:14:00",
                user=attacker_user,
                ip=attacker_ip,
                success=True,
                location="Eastern Europe",
                attempts=1,
            ),
        ]

        files = [
            FileAccessEvent(
                timestamp="2024-06-15T03:16:00",
                user=attacker_user,
                file_path="/data/payroll/salaries_2024.xlsx",
                action="read",
                sensitive=True,
            ),
            FileAccessEvent(
                timestamp="2024-06-15T03:17:00",
                user=attacker_user,
                file_path="/data/hr/employee_ssn.csv",
                action="copy",
                sensitive=True,
            ),
        ]

        alert = Alert(
            alert_id="ALERT-002",
            severity="medium",
            message="Login from unusual geographic location during off-hours",
            source_ip=attacker_ip,
            user=attacker_user,
            timestamp="2024-06-15T03:14:30",
        )

        return SystemState(
            step=0,
            task_id="medium",
            login_events=logins,
            file_events=files,
            alerts=[alert],
            sessions=[
                SessionData(session_id="S002", user=attacker_user, ip=attacker_ip,
                            login_time="2024-06-15T03:14:00", active=True,
                            privilege_level="normal"),
            ],
        )

    def _build_hard_scenario(self) -> SystemState:
        """
        Hard: Multi-step attack — login → privilege escalation → data exfiltration.
        """
        attacker_ip = "198.51.100.42"
        attacker_user = "eve"
        self._threat_ip = attacker_ip
        self._threat_user = attacker_user

        logins = [
            # Looks like a legit login at first
            LoginEvent(timestamp=self._ts(-30), user=attacker_user, ip=attacker_ip,
                       success=True, location="HQ-VPN", attempts=1),
            # Normal users
            LoginEvent(timestamp=self._ts(-60), user="frank", ip="10.0.0.20",
                       success=True, location="HQ", attempts=1),
        ]

        files = [
            # Reads sudoers file — suspicious but not alarming alone
            FileAccessEvent(timestamp=self._ts(-25), user=attacker_user,
                            file_path="/etc/sudoers", action="read", sensitive=True),
            # Writes a cron job backdoor
            FileAccessEvent(timestamp=self._ts(-20), user=attacker_user,
                            file_path="/etc/cron.d/backup_job", action="write", sensitive=True),
            # Reads sensitive data
            FileAccessEvent(timestamp=self._ts(-10), user=attacker_user,
                            file_path="/data/secrets/api_keys.json", action="read", sensitive=True),
        ]

        network = [
            # Large data transfer to external IP
            NetworkEvent(timestamp=self._ts(-5), src_ip=attacker_ip,
                         dst_ip="91.108.4.200", port=443, bytes_transferred=52_000_000,
                         protocol="HTTPS"),
        ]

        # Scattered low-severity alerts — agent must correlate them
        alerts = [
            Alert(alert_id="ALERT-003", severity="low",
                  message="Sensitive file /etc/sudoers accessed",
                  source_ip=attacker_ip, user=attacker_user, timestamp=self._ts(-25)),
            Alert(alert_id="ALERT-004", severity="low",
                  message="Cron configuration modified",
                  source_ip=attacker_ip, user=attacker_user, timestamp=self._ts(-20)),
            Alert(alert_id="ALERT-005", severity="medium",
                  message="Large outbound data transfer detected (52 MB)",
                  source_ip=attacker_ip, user=None, timestamp=self._ts(-5)),
        ]

        sessions = [
            SessionData(session_id="S003", user=attacker_user, ip=attacker_ip,
                        login_time=self._ts(-30), active=True, privilege_level="elevated"),
            SessionData(session_id="S004", user="frank", ip="10.0.0.20",
                        login_time=self._ts(-60), active=True, privilege_level="normal"),
        ]

        return SystemState(
            step=0,
            task_id="hard",
            login_events=logins,
            file_events=files,
            network_events=network,
            alerts=alerts,
            sessions=sessions,
        )

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _apply_action(self, action: str, target: Optional[str]) -> Tuple[float, Dict]:
        info: Dict[str, Any] = {"action": action, "target": target}
        reward = REWARD_NEUTRAL

        if action == "ignore":
            info["result"] = "No action taken."
            return reward, info

        if action == "flag_suspicious":
            if target and target == self._threat_ip:
                reward = REWARD_CORRECT_DETECTION if self._step_count <= self._early_detection_threshold \
                    else REWARD_LATE_DETECTION
                self._state.threat_detected = True
                self._state.flagged_ips.append(target)
                self._state.message = f"[✓] Correctly flagged threat IP {target}."
                info["result"] = "correct_flag"
            elif target:
                reward = REWARD_FALSE_POSITIVE
                self._state.flagged_ips.append(target)
                self._state.message = f"[✗] False positive — {target} is not the threat."
                info["result"] = "false_positive"
            else:
                info["result"] = "flag_no_target"

        elif action == "block_ip":
            if target and target == self._threat_ip:
                if self._step_count <= self._early_detection_threshold:
                    reward = REWARD_CORRECT_DETECTION
                else:
                    reward = REWARD_LATE_DETECTION
                self._state.threat_detected = True
                self._state.blocked_ips.append(target)
                self._state.message = f"[✓] Correctly blocked threat IP {target}."
                info["result"] = "correct_block"
            elif target:
                reward = REWARD_FALSE_POSITIVE
                self._state.blocked_ips.append(target)
                self._state.message = f"[✗] False positive — blocked legitimate IP {target}."
                info["result"] = "false_positive"
            else:
                info["result"] = "block_no_target"

        elif action == "escalate":
            if target and target in [a.alert_id for a in self._state.alerts
                                      if a.source_ip == self._threat_ip]:
                reward = REWARD_CORRECT_DETECTION if self._step_count <= self._early_detection_threshold \
                    else REWARD_LATE_DETECTION
                self._state.threat_detected = True
                self._state.escalated_alerts.append(target)
                self._state.message = f"[✓] Correctly escalated alert {target}."
                info["result"] = "correct_escalation"
            elif target:
                reward = REWARD_FALSE_POSITIVE
                self._state.escalated_alerts.append(target)
                self._state.message = f"[✗] Unnecessary escalation of {target}."
                info["result"] = "false_positive"
            else:
                info["result"] = "escalate_no_target"

        return reward, info

    def _advance_scenario(self):
        """Inject new events as the scenario unfolds (for hard task)."""
        if self.task_id != "hard":
            return
        # At step 3 we reveal privilege escalation attempt
        if self._step_count == 3:
            self._state.alerts.append(Alert(
                alert_id="ALERT-006",
                severity="high",
                message="Privilege escalation attempt: sudo -s executed by user 'eve'",
                source_ip=self._threat_ip,
                user=self._threat_user,
                timestamp=self._ts(5),
            ))
        # At step 5 reveal full exfil in network events label
        if self._step_count == 5:
            self._state.network_events.append(NetworkEvent(
                timestamp=self._ts(10),
                src_ip=self._threat_ip,
                dst_ip="91.108.4.200",
                port=443,
                bytes_transferred=105_000_000,
                protocol="HTTPS",
            ))
