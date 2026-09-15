"""Persistent synthetic executor. No robotics or model success is implied."""

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from roboweaver.contracts import (
    TERMINAL,
    ActionRecord,
    ActionStatus,
    ExecutionEvent,
    ExecutionSnapshot,
    Observation,
)


class MockExecutor:
    def __init__(self, path: Path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.connected = True
        self.lose_next_reply = False
        self.confirm_cancel = True
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS executions (key TEXT PRIMARY KEY, action TEXT, snapshot TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS world (id INTEGER PRIMARY KEY CHECK(id=1), state TEXT, version INTEGER)"
            )
            db.execute("INSERT OR IGNORE INTO world VALUES (1, '{}', 0)")

    def _db(self):
        # Short-lived connection context closes explicitly via SQLite subclass.
        class Connection(sqlite3.Connection):
            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    self.close()

        return sqlite3.connect(self.path, factory=Connection)

    def _online(self):
        if not self.connected:
            raise ConnectionError("Synthetic execution endpoint disconnected")

    async def submit(self, action: ActionRecord) -> ExecutionSnapshot:
        self._online()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT action,snapshot FROM executions WHERE key=?", (action.idempotency_key,)
            ).fetchone()
            if row:
                previous = ActionRecord.model_validate_json(row[0])
                if previous.proposal != action.proposal or previous.action_id != action.action_id:
                    raise ValueError("Idempotency key reused with different action")
                return ExecutionSnapshot.model_validate_json(row[1])
            snapshot = ExecutionSnapshot(
                execution_id=str(uuid4()),
                idempotency_key=action.idempotency_key,
                status=ActionStatus.RUNNING,
                stopped=False,
            )
            db.execute(
                "INSERT INTO executions VALUES (?,?,?)",
                (action.idempotency_key, action.model_dump_json(), snapshot.model_dump_json()),
            )
        if self.lose_next_reply:
            self.lose_next_reply = False
            raise ConnectionError("Synthetic submit reply lost after acceptance")
        return snapshot

    async def lookup(self, idempotency_key):
        self._online()
        with self._db() as db:
            row = db.execute(
                "SELECT snapshot FROM executions WHERE key=?", (idempotency_key,)
            ).fetchone()
        return ExecutionSnapshot.model_validate_json(row[0]) if row else None

    async def cancel(self, idempotency_key):
        self._online()
        snapshot = await self.lookup(idempotency_key)
        if snapshot is None:
            raise ValueError("Unknown execution")
        if snapshot.status not in TERMINAL and self.confirm_cancel:
            self.inject(idempotency_key, ActionStatus.CANCELED, stopped=True)
        return await self.lookup(idempotency_key)

    def inject(self, key, status=ActionStatus.COMPLETED, *, stopped=True, state=None):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT action,snapshot FROM executions WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown execution")
            action = ActionRecord.model_validate_json(row[0])
            snapshot = ExecutionSnapshot.model_validate_json(row[1])
            if snapshot.status in TERMINAL:
                raise ValueError("Execution is already terminal")
            event = ExecutionEvent(
                event_id=str(uuid4()),
                run_id=action.run_id,
                action_id=action.action_id,
                execution_id=snapshot.execution_id,
                plan_version=action.plan_version,
                sequence=snapshot.last_event.sequence + 1 if snapshot.last_event else 0,
                occurred_at=self.clock(),
                status=status,
                stopped=stopped,
                payload={"mode": "mock"},
            )
            snapshot.status, snapshot.stopped, snapshot.last_event = status, stopped, event
            db.execute(
                "UPDATE executions SET snapshot=? WHERE key=?", (snapshot.model_dump_json(), key)
            )
            if state is not None:
                db.execute(
                    "UPDATE world SET state=?,version=version+1 WHERE id=1", (json.dumps(state),)
                )
        return event

    async def observe(self):
        self._online()
        with self._db() as db:
            state, version = db.execute("SELECT state,version FROM world WHERE id=1").fetchone()
        return Observation(
            observation_id=str(uuid4()),
            observed_at=self.clock(),
            world_version=version,
            state=json.loads(state),
            source="mock",
        )

    @property
    def starts(self):
        with self._db() as db:
            return db.execute("SELECT count(*) FROM executions").fetchone()[0]
