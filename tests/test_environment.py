"""
PhantomShield X – Test Suite
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from environment.env import PhantomShieldEnv, SystemState, StepResult
from tasks.tasks import TaskRunner, TASK_REGISTRY
from graders.graders import get_grader, EasyGrader, MediumGrader, HardGrader


# ---------------------------------------------------------------------------
# Environment interface tests
# ---------------------------------------------------------------------------

class TestOpenEnvInterface:
    """Ensure reset / step / state adhere to the OpenEnv spec."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_reset_returns_system_state(self, task_id):
        env = PhantomShieldEnv(task_id=task_id)
        state = env.reset()
        assert isinstance(state, SystemState)
        assert state.step == 0
        assert state.done is False

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_state_after_reset(self, task_id):
        env = PhantomShieldEnv(task_id=task_id)
        env.reset()
        state = env.state()
        assert isinstance(state, SystemState)

    def test_step_before_reset_raises(self):
        env = PhantomShieldEnv(task_id="easy")
        with pytest.raises(RuntimeError):
            env.step("ignore")

    def test_state_before_reset_raises(self):
        env = PhantomShieldEnv(task_id="easy")
        with pytest.raises(RuntimeError):
            env.state()

    def test_invalid_action_raises(self):
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        with pytest.raises(ValueError):
            env.step("nuke_everything")

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_step_returns_step_result(self, task_id):
        env = PhantomShieldEnv(task_id=task_id)
        env.reset()
        result = env.step("ignore")
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.state, SystemState)

    def test_invalid_task_raises(self):
        with pytest.raises(AssertionError):
            PhantomShieldEnv(task_id="nightmare")


# ---------------------------------------------------------------------------
# Scenario content tests
# ---------------------------------------------------------------------------

class TestEasyScenario:
    def setup_method(self):
        self.env = PhantomShieldEnv(task_id="easy")
        self.state = self.env.reset()

    def test_has_login_events(self):
        assert len(self.state.login_events) > 0

    def test_has_alerts(self):
        assert len(self.state.alerts) > 0

    def test_threat_ip_in_logins(self):
        ips = {ev.ip for ev in self.state.login_events}
        assert "192.168.1.105" in ips

    def test_alert_severity_high(self):
        severities = {a.severity for a in self.state.alerts}
        assert "high" in severities


class TestMediumScenario:
    def setup_method(self):
        self.env = PhantomShieldEnv(task_id="medium")
        self.state = self.env.reset()

    def test_has_file_events(self):
        assert len(self.state.file_events) > 0

    def test_sensitive_files_accessed(self):
        sensitive = [e for e in self.state.file_events if e.sensitive]
        assert len(sensitive) >= 2

    def test_foreign_login_present(self):
        foreign = [ev for ev in self.state.login_events
                   if ev.location == "Eastern Europe"]
        assert len(foreign) >= 1


class TestMediumTermination:
    def test_medium_escalate_resolves_episode(self):
        env = PhantomShieldEnv(task_id="medium")
        env.reset()
        env.step("flag_suspicious", target="203.0.113.77")
        result = env.step("escalate", target="ALERT-002")
        assert result.done is True

    def test_medium_block_ip_resolves_episode(self):
        env = PhantomShieldEnv(task_id="medium")
        env.reset()
        result = env.step("block_ip", target="203.0.113.77")
        assert result.done is True


class TestHardScenario:
    def setup_method(self):
        self.env = PhantomShieldEnv(task_id="hard")
        self.state = self.env.reset()

    def test_has_network_events(self):
        assert len(self.state.network_events) > 0

    def test_large_transfer_present(self):
        large = [ev for ev in self.state.network_events if ev.bytes_transferred > 1_000_000]
        assert len(large) >= 1

    def test_multiple_alerts(self):
        assert len(self.state.alerts) >= 3

    def test_alert_006_appears_at_step_3(self):
        env = PhantomShieldEnv(task_id="hard")
        env.reset()
        for _ in range(3):
            result = env.step("ignore")
        alert_ids = {a.alert_id for a in result.state.alerts}
        assert "ALERT-006" in alert_ids


# ---------------------------------------------------------------------------
# Reward & action tests
# ---------------------------------------------------------------------------

class TestRewardLogic:
    def test_correct_block_gives_positive_reward(self):
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        result = env.step("block_ip", target="192.168.1.105")
        assert result.reward > 0

    def test_false_positive_gives_negative_reward(self):
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        result = env.step("block_ip", target="10.0.0.2")  # legitimate IP
        assert result.reward < 0

    def test_ignore_gives_zero_reward(self):
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        result = env.step("ignore")
        # reward is 0 unless it's the last step and threat is missed
        assert result.reward == 0.0 or result.reward == -1.0

    def test_early_detection_better_than_late(self):
        # Early
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        r_early = env.step("block_ip", target="192.168.1.105")

        # Late
        env2 = PhantomShieldEnv(task_id="easy")
        env2.reset()
        for _ in range(4):
            env2.step("ignore")
        r_late = env2.step("block_ip", target="192.168.1.105")

        assert r_early.reward >= r_late.reward

    def test_done_flag_set_on_detection(self):
        env = PhantomShieldEnv(task_id="easy")
        env.reset()
        result = env.step("block_ip", target="192.168.1.105")
        assert result.done is True

    def test_done_flag_on_max_steps(self):
        env = PhantomShieldEnv(task_id="easy")
        env._max_steps = 3
        env.reset()
        for _ in range(3):
            result = env.step("ignore")
        assert result.done is True


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:
    def _run_and_grade(self, task_id: str, actions):
        runner = TaskRunner(task_id)
        grader = get_grader(task_id)
        state = runner.reset()
        log = []
        cum_r = 0.0
        for step, (action, target) in enumerate(actions, 1):
            result = runner.step(action=action, target=target)
            cum_r += result.reward
            log.append({"step": step, "action": action, "target": target,
                         "reward": result.reward, "info": result.info})
            if result.done:
                break
        return grader.grade(result.state, cum_r, step, log)

    def test_easy_perfect_score(self):
        grade = self._run_and_grade("easy", [("block_ip", "192.168.1.105")])
        assert grade.score > 0.8

    def test_easy_false_positive_lowers_score(self):
        grade = self._run_and_grade(
            "easy",
            [("block_ip", "10.0.0.2"), ("block_ip", "192.168.1.105")]
        )
        perfect = self._run_and_grade("easy", [("block_ip", "192.168.1.105")])
        assert grade.score < perfect.score

    def test_easy_missed_threat_zero_ish(self):
        env = PhantomShieldEnv(task_id="easy")
        env._max_steps = 3
        grader = get_grader("easy")
        state = env.reset()
        log = []
        cum_r = 0.0
        for step in range(1, 4):
            result = env.step("ignore")
            cum_r += result.reward
            log.append({"step": step, "action": "ignore", "target": None,
                         "reward": result.reward, "info": result.info})
            if result.done:
                break
        grade = grader.grade(result.state, cum_r, step, log)
        assert grade.score < 0.4

    def test_score_always_in_range(self):
        for task_id in ["easy", "medium", "hard"]:
            grade = self._run_and_grade(task_id, [("ignore", None)] * 10)
            assert 0.0 <= grade.score <= 1.0

    def test_grade_result_has_required_fields(self):
        grade = self._run_and_grade("easy", [("ignore", None)])
        assert hasattr(grade, "score")
        assert hasattr(grade, "label")
        assert hasattr(grade, "breakdown")
        assert hasattr(grade, "feedback")
        assert isinstance(grade.breakdown, dict)

    def test_grader_is_deterministic(self):
        actions = [("block_ip", "192.168.1.105")]
        g1 = self._run_and_grade("easy", actions)
        g2 = self._run_and_grade("easy", actions)
        assert g1.score == g2.score


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_all_tasks_registered(self):
        assert "easy" in TASK_REGISTRY
        assert "medium" in TASK_REGISTRY
        assert "hard" in TASK_REGISTRY

    def test_task_runner_creates_env(self):
        runner = TaskRunner("easy")
        assert runner.env is not None

    def test_task_runner_invalid_raises(self):
        with pytest.raises(ValueError):
            TaskRunner("impossible")

    def test_describe_returns_string(self):
        runner = TaskRunner("medium")
        desc = runner.describe()
        assert isinstance(desc, str)
        assert "medium" in desc.lower() or "Medium" in desc
