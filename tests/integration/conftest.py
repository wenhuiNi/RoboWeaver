from pathlib import Path

import pytest

from roboweaver.adapters.mock import MockExecutor
from roboweaver.ledger import Ledger
from roboweaver.registry import ActionRegistry
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_task
from tests.support.clock import ManualClock


@pytest.fixture
async def rig(tmp_path):
    clock = ManualClock()
    executor = MockExecutor(tmp_path / "executor.db", clock)
    ledger = Ledger(tmp_path / "ledger.db")
    scheduler = Scheduler(ledger, executor, clock)
    fixtures = Path(__file__).parents[1] / "fixtures"
    task = load_task(fixtures / "task.json")
    run = scheduler.create(
        task,
        ActionRegistry.load(fixtures / "capabilities.json"),
        await executor.observe(),
        capacity=2,
    )
    return scheduler, executor, clock, run
