"""
PhantomShield X – Agent Graders
Deterministic graders that evaluate an agent's episode and return a score in [0.0, 1.0].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from environment.env import SystemState


@dataclass
class GradeResult:
    score: float                    # 0.0 – 1.0
    label: str                      # "Excellent" / "Good" / "Partial" / "Failed"
    breakdown: Dict[str, Any]       # per-criterion scores
    feedback: str                   # human-readable feedback


# ---------------------------------------------------------------------------
# Base grader
# ---------------------------------------------------------------------------

class BaseGrader:
    task_id: str
    threat_ip: str
    threat_user: Optional[str] = None

    def grade(
        self,
        final_state: SystemState,
        cumulative_reward: float,
        steps_taken: int,
        action_log: List[Dict],
    ) -> GradeResult:
        raise NotImplementedError

    @staticmethod
    def _normalize(raw: float, min_val: float, max_val: float) -> float:
        """Clamp and normalize raw score strictly to (0, 1)."""
        clamped = max(min_val, min(max_val, raw))
        normalized = (clamped - min_val) / (max_val - min_val)
        return max(0.01, min(0.99, normalized))

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.40:
            return "Partial"
        else:
            return "Failed"


# ---------------------------------------------------------------------------
# Easy grader
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Criteria:
    1. Was the threat IP correctly flagged or blocked?          (40 pts)
    2. Was detection early (within 3 steps)?                   (30 pts)
    3. No false positives committed?                           (20 pts)
    4. Minimal steps wasted on 'ignore'?                       (10 pts)
    """

    task_id = "easy"
    threat_ip = "192.168.1.105"

    def grade(self, final_state, cumulative_reward, steps_taken, action_log) -> GradeResult:
        breakdown: Dict[str, float] = {}

        # 1. Correct detection
        correctly_handled = (
            self.threat_ip in final_state.flagged_ips
            or self.threat_ip in final_state.blocked_ips
        )
        breakdown["correct_detection"] = 40.0 if correctly_handled else 0.0

        # 2. Early detection (within first 3 steps)
        detection_step = None
        for i, entry in enumerate(action_log, start=1):
            if entry.get("target") == self.threat_ip and entry.get("action") in (
                "flag_suspicious", "block_ip"
            ):
                detection_step = i
                break

        if detection_step is not None:
            if detection_step <= 3:
                breakdown["early_detection"] = 30.0
            elif detection_step <= 6:
                breakdown["early_detection"] = 15.0
            else:
                breakdown["early_detection"] = 5.0
        else:
            breakdown["early_detection"] = 0.0

        # 3. False positives (only reward if agent actually did something)
        fp_actions = [
            e for e in action_log
            if e.get("info", {}).get("result") == "false_positive"
        ]
        did_something = any(e.get("action") != "ignore" for e in action_log)
        if not correctly_handled and not did_something:
            breakdown["no_false_positives"] = 0.0   # passive agent gets no credit
        else:
            breakdown["no_false_positives"] = max(0.0, 20.0 - len(fp_actions) * 10.0)

        # 4. Efficiency (penalise fully passive runs)
        ignore_count = sum(1 for e in action_log if e.get("action") == "ignore")
        total_steps = len(action_log)
        # If agent did NOTHING useful (all ignores and never detected), give 0
        if not correctly_handled and ignore_count == total_steps:
            breakdown["efficiency"] = 0.0
        else:
            breakdown["efficiency"] = max(0.0, 10.0 - ignore_count * 2.0)

        raw_score = sum(breakdown.values())  # max 100
        score = round(raw_score / 100.0, 4)
        score = max(0.01, min(0.99, score))

        feedback_lines = [
            f"Threat IP correctly handled: {correctly_handled}",
            f"Detection at step: {detection_step or 'Never'}",
            f"False positives: {len(fp_actions)}",
            f"Ignore actions used: {ignore_count}",
            f"Raw score: {raw_score}/100",
        ]

        return GradeResult(
            score=score,
            label=self._label(score),
            breakdown=breakdown,
            feedback="\n".join(feedback_lines),
        )


# ---------------------------------------------------------------------------
# Medium grader
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Criteria:
    1. Correct IP/user identified                              (35 pts)
    2. Appropriate action taken (flag/block/escalate)          (25 pts)
    3. Alert ALERT-002 addressed                              (20 pts)
    4. No false positives                                      (10 pts)
    5. Detection before step 5                                 (10 pts)
    """

    task_id = "medium"
    threat_ip = "203.0.113.77"
    threat_user = "charlie"

    def grade(self, final_state, cumulative_reward, steps_taken, action_log) -> GradeResult:
        breakdown: Dict[str, float] = {}

        # 1. Correct identification
        identified = (
            self.threat_ip in final_state.flagged_ips
            or self.threat_ip in final_state.blocked_ips
            or "ALERT-002" in final_state.escalated_alerts
        )
        breakdown["correct_identification"] = 35.0 if identified else 0.0

        # 2. Appropriate action taken
        meaningful_actions = [
            e for e in action_log
            if e.get("action") in ("flag_suspicious", "block_ip", "escalate")
            and e.get("info", {}).get("result") in ("correct_flag", "correct_block", "correct_escalation")
        ]
        breakdown["appropriate_action"] = 25.0 if meaningful_actions else 0.0

        # 3. ALERT-002 addressed
        alert_addressed = "ALERT-002" in final_state.escalated_alerts
        breakdown["alert_addressed"] = 20.0 if alert_addressed else 0.0

        # 4. No false positives
        fp_count = sum(
            1 for e in action_log
            if e.get("info", {}).get("result") == "false_positive"
        )
        breakdown["no_false_positives"] = max(0.0, 10.0 - fp_count * 5.0)

        # 5. Speed
        detection_step = None
        for i, e in enumerate(action_log, 1):
            if e.get("info", {}).get("result") in (
                "correct_flag", "correct_block", "correct_escalation"
            ):
                detection_step = i
                break
        breakdown["speed"] = 10.0 if (detection_step and detection_step <= 5) else 0.0

        raw_score = sum(breakdown.values())
        score = round(max(0.01, min(0.99, raw_score / 100.0)), 4)

        feedback_lines = [
            f"Threat identified: {identified}",
            f"Alert ALERT-002 escalated: {alert_addressed}",
            f"Correct actions taken: {len(meaningful_actions)}",
            f"False positives: {fp_count}",
            f"Detection step: {detection_step or 'Never'}",
        ]

        return GradeResult(
            score=score,
            label=self._label(score),
            breakdown=breakdown,
            feedback="\n".join(feedback_lines),
        )


# ---------------------------------------------------------------------------
# Hard grader
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Criteria:
    1. Attack chain correlated (escalation issued)             (30 pts)
    2. Threat IP blocked before exfiltration completes         (25 pts)
    3. ALERT-006 (privilege escalation) recognized             (20 pts)
    4. No false positives                                      (15 pts)
    5. Acted within 6 steps of ALERT-006 appearing (step 3)   (10 pts)
    """

    task_id = "hard"
    threat_ip = "198.51.100.42"
    threat_user = "eve"

    def grade(self, final_state, cumulative_reward, steps_taken, action_log) -> GradeResult:
        breakdown: Dict[str, float] = {}

        # 1. Attack chain correlated — must have escalated something
        escalated = len(final_state.escalated_alerts) > 0
        breakdown["chain_correlated"] = 30.0 if escalated else 0.0

        # 2. Threat IP blocked
        blocked = self.threat_ip in final_state.blocked_ips
        breakdown["ip_blocked"] = 25.0 if blocked else 0.0

        # 3. ALERT-006 recognized (appears at step 3)
        alert006_escalated = "ALERT-006" in final_state.escalated_alerts
        breakdown["priv_esc_recognized"] = 20.0 if alert006_escalated else 0.0

        # 4. No false positives
        fp_count = sum(
            1 for e in action_log
            if e.get("info", {}).get("result") == "false_positive"
        )
        breakdown["no_false_positives"] = max(0.0, 15.0 - fp_count * 5.0)

        # 5. Speed after ALERT-006 (which appears at step 3)
        acted_after_006 = False
        for e in action_log:
            step_num = e.get("step", 999)
            if step_num >= 3 and e.get("action") in ("escalate", "block_ip"):
                if e.get("info", {}).get("result") in (
                    "correct_escalation", "correct_block"
                ):
                    if step_num <= 9:
                        acted_after_006 = True
                    break
        breakdown["response_speed"] = 10.0 if acted_after_006 else 0.0

        raw_score = sum(breakdown.values())
        score = round(max(0.01, min(0.99, raw_score / 100.0)), 4)

        feedback_lines = [
            f"Attack chain correlated (any escalation): {escalated}",
            f"Threat IP blocked: {blocked}",
            f"ALERT-006 (priv escalation) escalated: {alert006_escalated}",
            f"False positives: {fp_count}",
            f"Responded after ALERT-006 within 6 steps: {acted_after_006}",
        ]

        return GradeResult(
            score=score,
            label=self._label(score),
            breakdown=breakdown,
            feedback="\n".join(feedback_lines),
        )


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADER_REGISTRY:
        raise ValueError(f"No grader for task_id '{task_id}'")
    return GRADER_REGISTRY[task_id]
