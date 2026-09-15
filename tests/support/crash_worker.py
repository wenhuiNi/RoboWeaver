"""Hard process exits at integration boundaries; no production crash switches."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from roboweaver.adapters.scripted import ScriptedModel
from roboweaver.contracts import ExecutionEvent
from roboweaver.ledger import Ledger
from roboweaver.registry import ActionRegistry
from roboweaver.runtime import Runtime
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_task
from tests.support.remote_executor import RemoteExecutor


async def work(directory, socket, phase):
    executor = RemoteExecutor(socket)
    scheduler = Scheduler(Ledger(directory / "ledger.db"), executor)
    fixtures = Path(__file__).parents[1] / "fixtures"
    scenario = json.loads((fixtures / "scenario.json").read_text())
    if phase in {"next_pending", "revised"}:
        scenario["model_responses"].insert(1, scenario["model_responses"][0].copy())
    if phase == "resume":
        meta = json.loads((directory / "run.json").read_text())
        run_id, scenario = meta["run_id"], meta["scenario"]
    else:
        run_id = scheduler.create(
            load_task(fixtures / "task.json"),
            ActionRegistry.load(fixtures / "capabilities.json"),
            await executor.observe(),
        )
        (directory / "run.json").write_text(json.dumps({"run_id": run_id, "scenario": scenario}))
    model = ScriptedModel(responses=scenario["model_responses"])
    runtime = Runtime(scheduler, run_id, model, directory / "sessions.db")
    if phase == "accepted":
        submit = executor.submit

        async def crash_submit(action):
            await submit(action)
            os._exit(71)

        executor.submit = crash_submit
    try:
        if phase == "resume":
            session = await runtime._session()
            count = sum(
                bool(
                    e.content
                    and e.author == "robot_planner"
                    and any(p.function_call or p.text for p in e.content.parts or [])
                )
                for e in session.events
            )
            model.responses = scenario["model_responses"][count:]
            await runtime.resume()
            print(runtime.state.model_dump_json(), flush=True)
            return
        await runtime.start()
        action = runtime.state.actions[0]
        event = ExecutionEvent.model_validate(
            await executor.call(
                "inject",
                key=action.idempotency_key,
                status="FAILED" if phase == "revised" else "COMPLETED",
                state={"flag": phase != "revised"},
            )
        )
        if phase == "ledger":
            apply = scheduler.apply

            def crash_apply(event):
                apply(event)
                os._exit(72)

            scheduler.apply = crash_apply
        if phase == "outbox":

            async def crash_delivery():
                os._exit(73)

            runtime._deliver_feedback = crash_delivery
        if phase == "delivered":
            original = model.generate_content_async

            async def crash_model(request, stream=False):
                os._exit(74)
                async for response in original(request, stream):
                    yield response

            object.__setattr__(model, "generate_content_async", crash_model)
        if phase == "next_pending":
            drive = runtime._run

            async def crash_after_next(message=None, invocation_id=None):
                await drive(message, invocation_id)
                os._exit(75)

            runtime._run = crash_after_next
        if phase == "revised":
            recover = runtime.recover

            async def crash_after_revision():
                await recover()
                os._exit(76)

            runtime.recover = crash_after_revision
        await runtime.feedback(event)
    finally:
        await runtime.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["accepted", "ledger", "outbox", "delivered", "next_pending", "revised", "resume"],
        required=True,
    )
    args = parser.parse_args()
    asyncio.run(work(args.directory, args.socket, args.phase))
