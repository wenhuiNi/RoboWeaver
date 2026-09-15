"""Transport-independent task, action, observation, and execution contracts."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActionStatus(StrEnum):
    QUEUED = "QUEUED"
    SUBMITTING = "SUBMITTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMED_OUT = "TIMED_OUT"
    EXECUTION_UNKNOWN = "EXECUTION_UNKNOWN"
    INVALIDATED = "INVALIDATED"


TERMINAL = {
    ActionStatus.COMPLETED,
    ActionStatus.FAILED,
    ActionStatus.CANCELED,
    ActionStatus.TIMED_OUT,
    ActionStatus.INVALIDATED,
}


class TaskStatus(StrEnum):
    READY = "READY"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Budget(Contract):
    max_model_calls: int = Field(default=20, ge=1)
    max_replans: int = Field(default=3, ge=0)
    max_observations: int = Field(default=20, ge=1)
    max_api_retries: int = Field(default=2, ge=0)
    max_invalid_proposals: int = Field(default=2, ge=0)
    total_seconds: float = Field(default=300, gt=0)
    model_seconds: float = Field(default=20, gt=0)
    io_seconds: float = Field(default=5, gt=0)
    action_seconds: float = Field(default=30, gt=0)
    stop_seconds: float = Field(default=5, gt=0)
    observation_max_age: float = Field(default=5, gt=0)


class TaskSpec(Contract):
    task_type: str = Field(min_length=1)
    version: str = "1"
    goal: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    verifier: str = Field(min_length=3)
    budget: Budget = Field(default_factory=Budget)


class ActionContract(Contract):
    action_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1"
    description: str = Field(min_length=1)
    group: str | None = None
    parameters: dict[str, Any]
    constraints_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "additionalProperties": False}
    )
    resources: list[str] = Field(min_length=1)
    supports_cancel: bool
    required_state: dict[str, Any] = Field(default_factory=dict)
    completion_description: str = Field(min_length=1)


class Image(Contract):
    mime_type: Literal["image/png", "image/jpeg"]
    data_base64: str = Field(min_length=1)


class Observation(Contract):
    observation_id: str = Field(min_length=1)
    observed_at: float
    world_version: int = Field(ge=0)
    state: dict[str, Any] = Field(default_factory=dict)
    images: list[Image] = Field(default_factory=list)
    source: str = Field(min_length=1)
    valid: bool = True


class ActionProposal(Contract):
    action_type: str
    arguments: dict[str, Any]
    observation_id: str


class DecisionProposal(Contract):
    intent: Literal["actions", "observe", "finish"]
    actions: list[ActionProposal] = Field(default_factory=list, max_length=32)


class ActionRecord(Contract):
    run_id: str
    stream_id: str
    plan_version: int
    action_id: str
    sequence: int
    schema_version: str
    proposal: ActionProposal
    resources: list[str]
    idempotency_key: str
    timeout_seconds: float = Field(default=30, gt=0)
    status: ActionStatus = ActionStatus.QUEUED
    execution_id: str | None = None
    execution_observation_id: str | None = None
    submitted_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    stop_reason: str | None = None
    last_event_sequence: int = -1
    verified: bool = False


class ExecutionEvent(Contract):
    event_id: str
    run_id: str
    action_id: str
    execution_id: str
    plan_version: int
    sequence: int = Field(ge=0)
    occurred_at: float
    status: ActionStatus
    stopped: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(Contract):
    verdict: Verdict
    checkpoint: str
    observation_id: str
    observed_at: float
    reason: str
    source: str


class RunState(Contract):
    run_id: str
    stream_id: str
    task: TaskSpec
    observation: Observation
    capabilities: list[ActionContract]
    created_at: float
    capacity: int = Field(default=4, ge=1, le=32)
    plan_version: int = 1
    next_sequence: int = 1
    closed: bool = False
    frozen: bool = False
    status: TaskStatus = TaskStatus.READY
    actions: list[ActionRecord] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    binding: dict[str, str] = Field(default_factory=dict)
    pending_feedback: dict[str, Any] | None = None
    model_calls: int = 0
    api_retries: int = 0
    invalid_proposals: int = 0
    replans: int = 0
    observation_count: int = 1
    stop_intent: str | None = None
    reason: str | None = None


class ExecutionSnapshot(Contract):
    execution_id: str
    idempotency_key: str
    status: ActionStatus
    stopped: bool
    last_event: ExecutionEvent | None = None
