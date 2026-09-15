from pathlib import Path

import pytest

from roboweaver.registry import ActionRegistry
from roboweaver.tasks import load_task

pytestmark = pytest.mark.smoke


def test_s02_load_task_and_capabilities():
    fixtures = Path(__file__).parents[1] / "fixtures"
    registry = ActionRegistry.load(fixtures / "capabilities.json")
    task = load_task(fixtures / "task.json")
    registry.validate_task(task)
    assert registry.describe()[0]["parameters"]["type"] == "object"
    task.required_capabilities.append("not_registered")
    with pytest.raises(ValueError, match="Unavailable"):
        registry.validate_task(task)
