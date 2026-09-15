from pathlib import Path

import pytest
from jsonschema import ValidationError

from roboweaver.contracts import ActionProposal, Observation
from roboweaver.registry import ActionRegistry
from roboweaver.tasks import load_task, load_verifier

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_i02_task_capabilities_and_user_verifier():
    registry = ActionRegistry.load(FIXTURES / "capabilities.json")
    task = load_task(FIXTURES / "task.json")
    registry.validate_task(task)
    obs = Observation(
        observation_id="o1", observed_at=10, world_version=1, state={"flag": True}, source="mock"
    )
    assert load_verifier(task.verifier)(task, obs, "final").verdict == "PASS"
    assert set(registry.contracts) == {"set_flag", "open", "close"}
    assert "Open / Close" not in registry.contracts
    assert "pick" not in registry.contracts
    assert (
        registry.validate(
            ActionProposal(action_type="open", arguments={"target": "box"}, observation_id="o1"),
            obs,
            task,
            10,
        ).action_type
        == "open"
    )
    task.required_capabilities = ["pick"]
    with pytest.raises(ValueError, match="Unavailable"):
        registry.validate_task(task)


@pytest.mark.parametrize(
    "kind", ["unknown", "argument", "constraint", "stale", "observation", "precondition"]
)
def test_i02_invalid_actions_are_rejected(kind):
    registry = ActionRegistry.load(FIXTURES / "capabilities.json")
    task = load_task(FIXTURES / "task.json")
    obs = Observation(observation_id="o1", observed_at=10, world_version=1, source="mock")
    proposal = ActionProposal(
        action_type="set_flag", arguments={"value": True}, observation_id="o1"
    )
    now = 10
    if kind == "unknown":
        proposal.action_type = "Pick"
    if kind == "argument":
        proposal.arguments = {"value": "yes"}
    if kind == "constraint":
        task.constraints = {"force": 5}
    if kind == "stale":
        now = 100
    if kind == "observation":
        proposal.observation_id = "o0"
    if kind == "precondition":
        registry.contracts["set_flag"].required_state = {"ready": True}
    with pytest.raises((ValueError, ValidationError)):
        registry.validate(proposal, obs, task, now)


def test_i02_external_task_verifier_and_new_action(tmp_path, monkeypatch):
    import json

    from roboweaver.catalog import ACTION_GROUPS
    from roboweaver.contracts import ActionContract

    assert len(ACTION_GROUPS) == 21
    (tmp_path / "user_verifier.py").write_text(
        'def verify(task, observation, checkpoint):\n    return {"called": task.task_type}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    task_data = json.loads((FIXTURES / "task.json").read_text())
    task_data.update(
        task_type="new_user_task",
        required_capabilities=["user_action"],
        verifier="user_verifier:verify",
    )
    (tmp_path / "task.json").write_text(json.dumps(task_data))
    contract = json.loads((FIXTURES / "capabilities.json").read_text())[0]
    contract["action_type"] = "user_action"
    registry = ActionRegistry([ActionContract.model_validate(contract)])
    task = load_task(tmp_path / "task.json")
    registry.validate_task(task)
    assert load_verifier(task.verifier)(task, None, "final") == {"called": "new_user_task"}
