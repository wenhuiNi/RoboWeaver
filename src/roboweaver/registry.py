"""Only fully configured execution capabilities are visible to the planner."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from roboweaver.contracts import ActionContract, ActionProposal, Observation, TaskSpec


def local_schema(schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef"} and not item.startswith("#"):
                    raise ValueError("External schema references are not supported")
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(schema)
    return Draft202012Validator(schema)


class ActionRegistry:
    def __init__(self, contracts: list[ActionContract]):
        self.contracts = {}
        for contract in contracts:
            if contract.action_type in self.contracts:
                raise ValueError(f"Duplicate action type: {contract.action_type}")
            if contract.parameters.get("type") != "object":
                raise ValueError("Action parameters must be an object schema")
            local_schema(contract.parameters)
            local_schema(contract.constraints_schema)
            self.contracts[contract.action_type] = contract

    @classmethod
    def load(cls, path: Path):
        return cls([ActionContract.model_validate(item) for item in json.loads(path.read_text())])

    def validate_task(self, task: TaskSpec) -> None:
        missing = set(task.required_capabilities) - self.contracts.keys()
        if missing:
            raise ValueError(f"Unavailable capabilities: {sorted(missing)}")
        for name in task.required_capabilities:
            local_schema(self.contracts[name].constraints_schema).validate(task.constraints)

    def validate(
        self, proposal: ActionProposal, observation: Observation, task: TaskSpec, now: float
    ) -> ActionContract:
        if proposal.action_type not in self.contracts:
            raise ValueError(f"Unknown action type: {proposal.action_type}")
        if not observation.valid or proposal.observation_id != observation.observation_id:
            raise ValueError("Invalid or mismatched observation")
        if not 0 <= now - observation.observed_at <= task.budget.observation_max_age:
            raise ValueError("Stale or future observation")
        contract = self.contracts[proposal.action_type]
        local_schema(contract.parameters).validate(proposal.arguments)
        local_schema(contract.constraints_schema).validate(task.constraints)
        for key, expected in contract.required_state.items():
            if key not in observation.state or observation.state[key] != expected:
                raise ValueError(f"Unsatisfied precondition: {key}")
        return contract

    def describe(self) -> list[dict]:
        return [item.model_dump(mode="json") for item in self.contracts.values()]
