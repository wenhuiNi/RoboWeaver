"""Explicit synthetic scenario driver for the CLI; not a real robot adapter."""

import json
from pathlib import Path

from roboweaver.adapters.mock import MockExecutor
from roboweaver.adapters.scripted import ScriptedModel
from roboweaver.contracts import ActionStatus
from roboweaver.ledger import Ledger
from roboweaver.registry import ActionRegistry
from roboweaver.runtime import Runtime
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_task


def export_result(runtime, executor, directory):
    result = {
        "mode": "mock",
        "run_id": runtime.run_id,
        "status": runtime.state.status,
        "execution_starts": executor.starts,
        "model_calls": runtime.state.model_calls,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in runtime.scheduler.ledger.events(runtime.run_id))
    )
    return result


async def run_mock(task_path, capabilities_path, scenario_path, directory, until_waiting=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "run.json").exists():
        raise ValueError("Run directory already initialized; use a new directory or resume")
    scenario = json.loads(Path(scenario_path).read_text())
    model = ScriptedModel(responses=scenario["model_responses"])
    executor = MockExecutor(directory / "executor.db")
    scheduler = Scheduler(Ledger(directory / "ledger.db"), executor)
    run_id = scheduler.create(
        load_task(task_path), ActionRegistry.load(capabilities_path), await executor.observe()
    )
    (directory / "run.json").write_text(json.dumps({"run_id": run_id, "scenario": scenario}))
    runtime = Runtime(scheduler, run_id, model, directory / "sessions.db")
    try:
        await runtime.start()
        while not until_waiting and runtime.state.status == "EXECUTING":
            index = executor.starts - 1
            if index >= len(scenario.get("outcomes", [])):
                break
            outcome = scenario["outcomes"][index]
            action = runtime.state.actions[-1]
            event = executor.inject(
                action.idempotency_key,
                ActionStatus(outcome.get("status", "COMPLETED")),
                stopped=outcome.get("stopped", True),
                state=outcome.get("state"),
            )
            await runtime.feedback(event)
        return export_result(runtime, executor, directory)
    finally:
        await runtime.aclose()


async def cancel_mock(directory):
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    executor = MockExecutor(directory / "executor.db")
    scheduler = Scheduler(Ledger(directory / "ledger.db"), executor)
    state = scheduler.ledger.read(meta["run_id"])
    if state.status not in {"SUCCEEDED", "CANCELED", "FAILED"}:
        await scheduler.request_stop(state.run_id, "cancel")
        state = await scheduler.tick(state.run_id)
    return {"mode": "mock", "run_id": state.run_id, "status": state.status}
