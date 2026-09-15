"""SQLite run state, resource ownership, and append-only domain journal."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from roboweaver.contracts import RunState


class Ledger:
    def __init__(self, path: Path, *, create: bool = True):
        self.path = Path(path)
        if not create:
            if not self.path.is_file():
                raise ValueError("Run ledger does not exist")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS resources (name TEXT PRIMARY KEY, owner TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS journal (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS received (
                    run_id TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(run_id, event_id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, state: RunState):
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?)", (state.run_id, state.model_dump_json()))
            self.log(
                db, state.run_id, "run_created", {"mode": "mock", "task": state.task.model_dump()}
            )

    def read(self, run_id: str) -> RunState:
        with self.connect() as db:
            row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown run: {run_id}")
        return RunState.model_validate_json(row[0])

    @contextmanager
    def transaction(self, run_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown run: {run_id}")
            state = RunState.model_validate_json(row[0])
            yield db, state
            db.execute("UPDATE runs SET state=? WHERE id=?", (state.model_dump_json(), run_id))

    @staticmethod
    def log(db, run_id, kind, payload):
        db.execute(
            "INSERT INTO journal(run_id,kind,payload) VALUES(?,?,?)",
            (run_id, kind, json.dumps(payload)),
        )

    def events(self, run_id):
        with self.connect() as db:
            return [
                {"sequence": seq, "type": kind, "payload": json.loads(payload)}
                for seq, kind, payload in db.execute(
                    "SELECT sequence,kind,payload FROM journal WHERE run_id=? ORDER BY sequence",
                    (run_id,),
                )
            ]

    @contextmanager
    def claim(self, run_id):
        """One active agent driver per run; the OS releases this lock after a crash."""
        import fcntl
        from uuid import UUID

        # IDs become filenames only after validation.
        filename = f"{UUID(run_id)}.lock"
        with (self.path.parent / filename).open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunBusyError("Another process is driving this run") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class RunBusyError(RuntimeError):
    pass
