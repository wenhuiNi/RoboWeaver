import pytest

from roboweaver.contracts import ActionStatus
from tests.integration.test_runtime import make_runtime, script
from tests.integration.test_scheduler import proposal

pytestmark = pytest.mark.integration


async def test_i08_failure_replans_with_new_observation(tmp_path):
    runtime, executor, clock = await make_runtime(tmp_path, script(True, True))
    try:
        await runtime.start()
        old = runtime.state.actions[0]
        clock.advance(1)
        await runtime.feedback(
            executor.inject(old.idempotency_key, ActionStatus.FAILED, state={"flag": False})
        )
        new = runtime.state.actions[1]
        assert new.plan_version == 2 and new.proposal.observation_id != old.proposal.observation_id
        assert executor.starts == 2
        clock.advance(1)
        await runtime.feedback(executor.inject(new.idempotency_key, state={"flag": True}))
        assert runtime.state.status == "SUCCEEDED" and runtime.state.replans == 1
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("failure", [ActionStatus.FAILED, ActionStatus.COMPLETED])
async def test_i08_repeated_failure_or_unknown_exhausts_budget(tmp_path, failure):
    runtime, executor, clock = await make_runtime(tmp_path, script(True, True, True))
    with runtime.scheduler.ledger.transaction(runtime.run_id) as (db, state):
        state.task.budget.max_replans = 1
    try:
        await runtime.start()
        for _ in range(2):
            clock.advance(1)
            action = runtime.state.actions[-1]
            await runtime.feedback(executor.inject(action.idempotency_key, failure, state={}))
        assert runtime.state.status == "FAILED"
        assert runtime.state.replans == 1 and executor.starts == 2
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("error", ["timeout", "rate_limit"])
async def test_i11_transient_api_retry_does_not_retry_robot(tmp_path, error):
    runtime, executor, clock = await make_runtime(tmp_path, [{"error": error}, *script(True)])
    try:
        await runtime.start()
        assert runtime.state.api_retries == 1 and runtime.state.model_calls == 2
        assert executor.starts == 1
    finally:
        await runtime.aclose()


async def test_i11_invalid_decision_can_be_repaired(tmp_path):
    invalid = {
        "tool": "execute_action",
        "arguments": {
            "action_type": "unknown",
            "arguments": {},
            "observation_id": "$observation_id",
        },
    }
    runtime, executor, clock = await make_runtime(tmp_path, [invalid, *script(True)])
    try:
        await runtime.start()
        assert executor.starts == 1
        assert runtime.state.invalid_proposals == 1 and runtime.state.model_calls == 2
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("budget", ["max_model_calls", "max_observations"])
async def test_i08_budget_failure_is_bounded(tmp_path, budget):
    runtime, executor, clock = await make_runtime(tmp_path, script(True))
    with runtime.scheduler.ledger.transaction(runtime.run_id) as (db, state):
        setattr(state.task.budget, budget, 1)
    try:
        await runtime.start()
        a = runtime.state.actions[0]
        clock.advance(1)
        await runtime.feedback(executor.inject(a.idempotency_key, state={"flag": True}))
        assert runtime.state.status == "FAILED" and executor.starts == 1
    finally:
        await runtime.aclose()


async def test_i09_cancel_unconfirmed_locks_other_runs(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    await s.dispatch(run)
    e.confirm_cancel = False
    await s.request_stop(run, "cancel")
    c.advance(6)
    state = await s.tick(run)
    assert state.status == "BLOCKED" and state.actions[0].status == "EXECUTION_UNKNOWN"
    from roboweaver.registry import ActionRegistry

    other = s.create(state.task, ActionRegistry(state.capabilities), await e.observe())
    s.append(other, 1, [proposal(s, other)])
    with pytest.raises(ValueError, match="occupied"):
        await s.dispatch(other)
    e.confirm_cancel = True
    await s.request_stop(run, "cancel")
    assert (await s.tick(run)).status == "CANCELED"
    await s.dispatch(other)
    assert e.starts == 2


async def test_i09_timeout_without_model_call(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    await s.dispatch(run)
    c.advance(31)
    state = await s.tick(run)
    assert state.status == "FAILED" and state.stop_intent == "timeout"
    assert state.actions[0].status in {"CANCELED", "TIMED_OUT"}
    assert state.model_calls == 0


async def test_i09_disconnect_and_late_stop(rig):
    s, e, c, run = rig
    s.append(run, 1, [proposal(s, run)])
    await s.dispatch(run)
    e.connected = False
    await s.request_stop(run, "cancel")
    assert s.ledger.read(run).status == "BLOCKED"
    e.connected = True
    a = s.ledger.read(run).actions[0]
    e.inject(a.idempotency_key, ActionStatus.CANCELED)
    assert (await s.tick(run)).status == "CANCELED"


async def test_i11_suspended_model_has_real_timeout(tmp_path):
    import asyncio

    from roboweaver.adapters.scripted import ScriptedModel
    from roboweaver.agent.model import BudgetedModel

    class NeverReturns(ScriptedModel):
        async def generate_content_async(self, llm_request, stream=False):
            await asyncio.Event().wait()
            if False:
                yield

    runtime, executor, clock = await make_runtime(tmp_path, [])
    slow = NeverReturns()
    runtime.runner.agent.model = BudgetedModel(delegate=slow, runtime=runtime)
    with runtime.scheduler.ledger.transaction(runtime.run_id) as (db, state):
        state.task.budget.model_seconds = 0.02
        state.task.budget.max_api_retries = 1
    try:
        await runtime.start()
        assert runtime.state.status == "FAILED"
        assert runtime.state.model_calls == 2 and executor.starts == 0
    finally:
        await runtime.aclose()


def test_i09_cli_cancel(tmp_path):
    import json
    from pathlib import Path

    from tests.smoke.test_cli import cli

    fixtures = Path(__file__).parents[1] / "fixtures"
    result = cli(
        "run",
        "--mode",
        "mock",
        "--task",
        str(fixtures / "task.json"),
        "--capabilities",
        str(fixtures / "capabilities.json"),
        "--scenario",
        str(fixtures / "scenario.json"),
        "--workdir",
        str(tmp_path),
        "--until-waiting",
    )
    assert result.returncode == 1 and json.loads(result.stdout)["status"] == "EXECUTING"
    result = cli("cancel", "--workdir", str(tmp_path))
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "CANCELED"
