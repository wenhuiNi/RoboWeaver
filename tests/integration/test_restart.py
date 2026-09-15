import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from roboweaver.ledger import Ledger
from tests.support.remote_executor import RemoteExecutor

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "phase,code",
    [
        ("accepted", 71),
        ("ledger", 72),
        ("outbox", 73),
        ("delivered", 74),
        ("next_pending", 75),
        ("revised", 76),
    ],
)
async def test_i10_hard_crash_with_independent_executor(tmp_path, phase, code):
    socket = tmp_path / "executor.sock"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support.executor_server",
            "--socket",
            str(socket),
            "--database",
            str(tmp_path / "executor.db"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = await asyncio.wait_for(asyncio.to_thread(server.stdout.readline), timeout=10)
        assert ready.strip() == "READY"

        def command(step):
            return [
                sys.executable,
                "-m",
                "tests.support.crash_worker",
                "--directory",
                str(tmp_path),
                "--socket",
                str(socket),
                "--phase",
                step,
            ]

        process = await asyncio.create_subprocess_exec(
            *command(phase), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        assert process.returncode == code, stderr.decode()
        run = json.loads((tmp_path / "run.json").read_text())["run_id"]
        state = Ledger(tmp_path / "ledger.db").read(run)
        executor = RemoteExecutor(socket)
        expected_starts = 2 if phase == "next_pending" else 1
        assert await executor.call("starts") == expected_starts
        if phase in {"accepted", "next_pending"}:
            if phase == "accepted":
                assert state.actions[0].execution_id is None
            await executor.call(
                "inject", key=state.actions[-1].idempotency_key, state={"flag": True}
            )
        assert server.poll() is None
        for attempt in range(3 if phase == "revised" else 2):
            process = await asyncio.create_subprocess_exec(
                *command("resume"), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
            assert process.returncode == 0, stderr.decode()
            result = json.loads(stdout)
            if phase == "revised" and attempt == 0:
                assert result["status"] == "EXECUTING" and result["replans"] == 1
                expected_starts = 2
                await executor.call(
                    "inject", key=result["actions"][-1]["idempotency_key"], state={"flag": True}
                )
            else:
                assert result["status"] == "SUCCEEDED", stdout.decode()
            assert await executor.call("starts") == expected_starts
        assert server.poll() is None
    finally:
        server.terminate()
        await asyncio.to_thread(server.wait, 5)
        server.stdout.close()
        server.stderr.close()


def test_i10_cli_resume_waiting_run(tmp_path):
    from tests.smoke.test_cli import cli

    fixtures = Path(__file__).parents[1] / "fixtures"
    result = cli(
        "run",
        "--mode",
        "mock",
        "--task",
        str(fixtures / "task.json"),
        "--capabilities",
        str(fixtures / "capabilities.json"),
        "--scenario",
        str(fixtures / "scenario.json"),
        "--workdir",
        str(tmp_path),
        "--until-waiting",
    )
    assert result.returncode == 1
    result = cli("resume", "--workdir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["execution_starts"] == 1
