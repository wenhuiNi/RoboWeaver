import base64
from pathlib import Path

import pytest

from roboweaver.adapters.mock import MockExecutor
from roboweaver.adapters.scripted import ScriptedModel
from roboweaver.contracts import Image
from roboweaver.ledger import Ledger
from roboweaver.registry import ActionRegistry
from roboweaver.runtime import Runtime
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_task
from tests.support.clock import ManualClock

pytestmark = pytest.mark.integration


def script(*values):
    return [
        {
            "tool": "execute_action",
            "arguments": {
                "action_type": "set_flag",
                "arguments": {"value": v},
                "observation_id": "$observation_id",
            },
        }
        for v in values
    ] + [{"tool": "finish_task"}, {"text": "Finished"}]


async def make_runtime(tmp_path, responses):
    clock = ManualClock()
    executor = MockExecutor(tmp_path / "executor.db", clock)
    scheduler = Scheduler(Ledger(tmp_path / "ledger.db"), executor, clock)
    fixtures = Path(__file__).parents[1] / "fixtures"
    obs = await executor.observe()
    obs.images = [
        Image(
            mime_type="image/png",
            data_base64=base64.b64encode(b"synthetic-image-transport").decode(),
        )
    ]
    run = scheduler.create(
        load_task(fixtures / "task.json"), ActionRegistry.load(fixtures / "capabilities.json"), obs
    )
    model = ScriptedModel(responses=responses)
    return Runtime(scheduler, run, model, tmp_path / "sessions.db", clock), executor, clock


async def test_i01_i12_real_adk_wait_feedback_and_multistep(tmp_path):
    runtime, executor, clock = await make_runtime(tmp_path, script(True, True))
    try:
        await runtime.start()
        assert executor.starts == 1 and runtime.state.model_calls == 1
        request = runtime.model.requests[0]
        assert any(
            p.inline_data and p.inline_data.data == b"synthetic-image-transport"
            for c in request.contents
            for p in c.parts
        )
        assert "set_flag" in str(request.contents)
        await runtime.continue_execution()
        assert runtime.state.model_calls == 1
        for count in [1, 2]:
            a = runtime.state.actions[-1]
            clock.advance(1)
            await runtime.feedback(executor.inject(a.idempotency_key, state={"flag": True}))
            assert executor.starts == min(count + 1, 2)
        assert runtime.state.status == "SUCCEEDED"
        assert len(runtime.state.verifications) == 3
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("state", [{"flag": False}, {}])
async def test_i07_completion_is_not_task_success(tmp_path, state):
    runtime, executor, clock = await make_runtime(tmp_path, script(True))
    try:
        await runtime.start()
        a = runtime.state.actions[-1]
        clock.advance(1)
        await runtime.feedback(executor.inject(a.idempotency_key, state=state))
        assert runtime.state.status != "SUCCEEDED"
        assert runtime.state.verifications[0].verdict in {"FAIL", "UNKNOWN"}
    finally:
        await runtime.aclose()


@pytest.mark.parametrize(
    "response",
    [
        {"error": "timeout"},
        {
            "tool": "execute_action",
            "arguments": {
                "action_type": "unknown",
                "arguments": {},
                "observation_id": "$observation_id",
            },
        },
    ],
)
async def test_i11_model_errors_do_not_dispatch(tmp_path, response):
    runtime, executor, clock = await make_runtime(tmp_path, [response])
    try:
        await runtime.start()
        assert executor.starts == 0
        assert runtime.state.status == "FAILED"
    finally:
        await runtime.aclose()
