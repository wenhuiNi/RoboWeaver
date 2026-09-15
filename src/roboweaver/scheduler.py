"""Bounded serial action streams with durable intent and execution reconciliation."""

import asyncio
import json
import sqlite3
import time
from uuid import uuid4

from roboweaver.contracts import (
    TERMINAL,
    ActionProposal,
    ActionRecord,
    ActionStatus,
    ExecutionEvent,
    Observation,
    RunState,
    TaskSpec,
    TaskStatus,
    Verdict,
    VerificationResult,
)
from roboweaver.ledger import Ledger
from roboweaver.registry import ActionRegistry


class Scheduler:
    def __init__(self, ledger: Ledger, executor, clock=time.time):
        self.ledger, self.executor, self.clock = ledger, executor, clock

    def create(
        self, task: TaskSpec, registry: ActionRegistry, observation: Observation, capacity=4
    ):
        registry.validate_task(task)
        run_id = str(uuid4())
        self.ledger.create(
            RunState(
                run_id=run_id,
                stream_id=str(uuid4()),
                task=task,
                observation=observation,
                capabilities=list(registry.contracts.values()),
                capacity=capacity,
                created_at=self.clock(),
            )
        )
        return run_id

    @staticmethod
    def _version(state, version):
        if version != state.plan_version:
            raise ValueError("Plan version conflict")

    def append(self, run_id, version, proposals: list[ActionProposal], *, binding=None):
        if not proposals:
            raise ValueError("An append must contain actions")
        with self.ledger.transaction(run_id) as (db, state):
            self._version(state, version)
            if (
                state.closed
                or state.frozen
                or state.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}
            ):
                raise ValueError("Stream does not accept actions")
            pending = sum(a.status not in TERMINAL for a in state.actions)
            if pending + len(proposals) > state.capacity:
                raise ValueError("Action stream capacity exceeded")
            registry = ActionRegistry(state.capabilities)
            for proposal in proposals:
                contract = registry.validate(proposal, state.observation, state.task, self.clock())
                action_id = str(uuid4())
                state.actions.append(
                    ActionRecord(
                        run_id=run_id,
                        stream_id=state.stream_id,
                        plan_version=version,
                        action_id=action_id,
                        sequence=state.next_sequence,
                        schema_version=contract.version,
                        proposal=proposal,
                        resources=contract.resources,
                        idempotency_key=action_id,
                        timeout_seconds=state.task.budget.action_seconds,
                    )
                )
                state.next_sequence += 1
            if binding is not None:
                state.binding = {**binding, "action_id": state.actions[-1].action_id}
            self.ledger.log(
                db, run_id, "actions_accepted", {"plan_version": version, "count": len(proposals)}
            )
        return self.ledger.read(run_id).next_sequence - 1

    def close(self, run_id, version, last_sequence):
        with self.ledger.transaction(run_id) as (db, state):
            self._version(state, version)
            if state.frozen or last_sequence != state.next_sequence - 1:
                raise ValueError("Invalid final sequence or frozen stream")
            state.closed = True
            self.ledger.log(db, run_id, "stream_closed", {"last_sequence": last_sequence})

    async def dispatch(self, run_id):
        with self.ledger.transaction(run_id) as (db, state):
            if state.frozen or state.status in {
                TaskStatus.FAILED,
                TaskStatus.CANCELED,
                TaskStatus.SUCCEEDED,
                TaskStatus.BLOCKED,
            }:
                return None
            current = [
                a
                for a in state.actions
                if a.plan_version == state.plan_version and a.status != ActionStatus.INVALIDATED
            ]
            action = next(
                (a for a in current if a.status != ActionStatus.COMPLETED or not a.verified), None
            )
            if action is None or action.status != ActionStatus.QUEUED:
                return None
            registry = ActionRegistry(state.capabilities)
            registry.validate(
                action.proposal.model_copy(
                    update={"observation_id": state.observation.observation_id}
                ),
                state.observation,
                state.task,
                self.clock(),
            )
            action.execution_observation_id = state.observation.observation_id
            try:
                for resource in action.resources:
                    db.execute("INSERT INTO resources VALUES (?,?)", (resource, action.action_id))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Execution resource is occupied") from exc
            action.status, action.submitted_at = ActionStatus.SUBMITTING, self.clock()
            state.status = TaskStatus.EXECUTING
            self.ledger.log(db, run_id, "submit_intent", action.model_dump(mode="json"))
        try:
            async with asyncio.timeout(state.task.budget.io_seconds):
                snapshot = await self.executor.submit(action)
        except (ConnectionError, TimeoutError):
            self._unknown(run_id, action.action_id)
            return None
        self._accept_snapshot(run_id, action.action_id, snapshot)
        return snapshot.execution_id

    def _unknown(self, run_id, action_id):
        with self.ledger.transaction(run_id) as (db, state):
            action = next(a for a in state.actions if a.action_id == action_id)
            if action.status not in TERMINAL:
                action.status = ActionStatus.EXECUTION_UNKNOWN
                state.status = TaskStatus.BLOCKED
                self.ledger.log(db, run_id, "execution_unknown", {"action_id": action_id})

    def _accept_snapshot(self, run_id, action_id, snapshot):
        with self.ledger.transaction(run_id) as (db, state):
            action = next(a for a in state.actions if a.action_id == action_id)
            if snapshot.idempotency_key != action.idempotency_key:
                raise ValueError("Execution snapshot key mismatch")
            if action.execution_id and action.execution_id != snapshot.execution_id:
                raise ValueError("Execution ID changed")
            action.execution_id = snapshot.execution_id
            if action.status not in TERMINAL:
                if snapshot.status in TERMINAL and not snapshot.last_event:
                    raise ValueError("Terminal snapshot requires execution evidence")
                if snapshot.status not in TERMINAL:
                    action.status = (
                        ActionStatus.STOPPING
                        if action.stop_requested_at is not None
                        else ActionStatus.RUNNING
                    )
                if state.status == TaskStatus.BLOCKED and snapshot.status not in TERMINAL:
                    state.status = TaskStatus.EXECUTING
        if snapshot.last_event:
            self.apply(snapshot.last_event)

    async def reconcile(self, run_id):
        for action in self.ledger.read(run_id).actions:
            if action.status in TERMINAL or action.status == ActionStatus.QUEUED:
                continue
            try:
                async with asyncio.timeout(self.ledger.read(run_id).task.budget.io_seconds):
                    snapshot = await self.executor.lookup(action.idempotency_key)
                if snapshot is None:
                    self._unknown(run_id, action.action_id)
                else:
                    self._accept_snapshot(run_id, action.action_id, snapshot)
            except (ConnectionError, TimeoutError):
                self._unknown(run_id, action.action_id)

    def apply(self, event: ExecutionEvent):
        with self.ledger.transaction(event.run_id) as (db, state):
            action = next((a for a in state.actions if a.action_id == event.action_id), None)
            if (
                action is None
                or action.plan_version != event.plan_version
                or action.execution_id != event.execution_id
            ):
                raise ValueError("Execution event association mismatch")
            payload = event.model_dump_json()
            old = db.execute(
                "SELECT payload FROM received WHERE run_id=? AND event_id=?",
                (event.run_id, event.event_id),
            ).fetchone()
            if old:
                if json.loads(old[0]) != json.loads(payload):
                    raise ValueError("Conflicting duplicate event")
                return
            db.execute(
                "INSERT INTO received VALUES (?,?,?)", (event.run_id, event.event_id, payload)
            )
            self.ledger.log(db, event.run_id, "execution_event", event.model_dump(mode="json"))
            if event.sequence <= action.last_event_sequence or action.status in TERMINAL:
                return
            action.last_event_sequence = event.sequence
            if event.status in TERMINAL:
                if not event.stopped or event.status == ActionStatus.INVALIDATED:
                    action.status = ActionStatus.EXECUTION_UNKNOWN
                    state.status = TaskStatus.BLOCKED
                    return
                action.status, action.finished_at = event.status, event.occurred_at
                db.execute("DELETE FROM resources WHERE owner=?", (action.action_id,))
                if action.plan_version == state.plan_version:
                    state.status = (
                        TaskStatus.VERIFYING
                        if event.status == ActionStatus.COMPLETED
                        else TaskStatus.RECOVERING
                    )
            elif event.status == ActionStatus.RUNNING:
                action.status = (
                    ActionStatus.STOPPING
                    if action.stop_requested_at is not None
                    else ActionStatus.RUNNING
                )

    async def request_stop(self, run_id, reason="cancel"):
        with self.ledger.transaction(run_id) as (db, state):
            state.frozen = True
            state.stop_intent = reason
            for action in state.actions:
                if action.status == ActionStatus.QUEUED:
                    action.status = ActionStatus.INVALIDATED
                elif action.status not in TERMINAL:
                    action.status = ActionStatus.STOPPING
                    if action.stop_requested_at is None:
                        action.stop_requested_at = self.clock()
                    action.stop_reason = reason
            self.ledger.log(db, run_id, "stop_requested", {"reason": reason})
        for action in self.ledger.read(run_id).actions:
            if action.status != ActionStatus.STOPPING:
                continue
            try:
                contract = next(
                    c
                    for c in self.ledger.read(run_id).capabilities
                    if c.action_type == action.proposal.action_type
                )
                if not contract.supports_cancel:
                    self._unknown(run_id, action.action_id)
                    continue
                async with asyncio.timeout(self.ledger.read(run_id).task.budget.io_seconds):
                    snapshot = await self.executor.cancel(action.idempotency_key)
                self._accept_snapshot(run_id, action.action_id, snapshot)
            except (ConnectionError, TimeoutError, ValueError):
                self._unknown(run_id, action.action_id)

    def revise(self, run_id, expected_version, observation):
        with self.ledger.transaction(run_id) as (db, state):
            self._version(state, expected_version)
            if not state.frozen:
                raise ValueError("Freeze and reconcile the old stream before revision")
            if any(a.status not in TERMINAL for a in state.actions):
                raise ValueError("Old execution is not confirmed stopped")
            if not observation.valid or observation.observed_at < max(
                [a.finished_at or state.created_at for a in state.actions], default=state.created_at
            ):
                raise ValueError("Revision requires a fresh observation")
            state.plan_version += 1
            state.next_sequence = 1
            state.closed = state.frozen = False
            state.stop_intent = None
            state.observation = observation
            state.status = TaskStatus.READY
            self.ledger.log(db, run_id, "stream_revised", {"plan_version": state.plan_version})
        return self.ledger.read(run_id).plan_version

    def verify(self, run_id, result: VerificationResult):
        with self.ledger.transaction(run_id) as (db, state):
            if (
                result.observation_id != state.observation.observation_id
                or result.observed_at != state.observation.observed_at
            ):
                raise ValueError("Verification evidence does not match current observation")
            if (
                not state.observation.valid
                or not 0
                <= self.clock() - result.observed_at
                <= state.task.budget.observation_max_age
            ):
                raise ValueError("Verification requires fresh valid evidence")
            current = [
                a
                for a in state.actions
                if a.plan_version == state.plan_version and a.status != ActionStatus.INVALIDATED
            ]
            finished = [a for a in current if a.status == ActionStatus.COMPLETED and not a.verified]
            if result.source != state.observation.source:
                raise ValueError("Verification evidence source mismatch")
            if any(
                result.observation_id == a.execution_observation_id
                or result.observed_at < (a.finished_at or 0)
                for a in finished
            ):
                raise ValueError("Verification predates execution completion")
            state.verifications.append(result)
            if result.verdict == Verdict.PASS:
                if result.checkpoint != "final":
                    action = next((a for a in finished if a.action_id == result.checkpoint), None)
                    if action is None:
                        raise ValueError("Unknown verification checkpoint")
                    action.verified = True
                elif (
                    state.closed
                    and not state.frozen
                    and all(a.status == ActionStatus.COMPLETED and a.verified for a in current)
                ):
                    state.status = TaskStatus.SUCCEEDED
            else:
                state.status = TaskStatus.RECOVERING
            self.ledger.log(db, run_id, "verification", result.model_dump(mode="json"))

    def observe(self, run_id, observation):
        with self.ledger.transaction(run_id) as (db, state):
            if (
                observation.world_version < state.observation.world_version
                or observation.observed_at < state.observation.observed_at
            ):
                raise ValueError("Observation moved backwards")
            state.observation = observation
            self.ledger.log(db, run_id, "observation", observation.model_dump(mode="json"))

    async def tick(self, run_id):
        state = self.ledger.read(run_id)
        if state.status in {TaskStatus.SUCCEEDED, TaskStatus.CANCELED, TaskStatus.FAILED}:
            return state
        active = [
            a for a in state.actions if a.status not in TERMINAL and a.status != ActionStatus.QUEUED
        ]
        total_expired = self.clock() - state.created_at >= state.task.budget.total_seconds
        action_expired = any(
            a.submitted_at is not None and self.clock() - a.submitted_at >= a.timeout_seconds
            for a in active
        )
        if not state.stop_intent and (total_expired or action_expired):
            await self.request_stop(run_id, "timeout")
        await self.reconcile(run_id)
        with self.ledger.transaction(run_id) as (db, state):
            for action in state.actions:
                if (
                    action.status not in TERMINAL
                    and action.stop_requested_at is not None
                    and self.clock() - action.stop_requested_at >= state.task.budget.stop_seconds
                ):
                    action.status = ActionStatus.EXECUTION_UNKNOWN
                    state.status = TaskStatus.BLOCKED
            if state.stop_intent and all(a.status in TERMINAL for a in state.actions):
                state.status = (
                    TaskStatus.CANCELED
                    if state.stop_intent == "cancel"
                    else TaskStatus.RECOVERING
                    if state.stop_intent == "replan"
                    else TaskStatus.FAILED
                )
        return self.ledger.read(run_id)
