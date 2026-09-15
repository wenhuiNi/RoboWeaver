import json
from pathlib import Path

import pytest

from tests.smoke.test_cli import cli

pytestmark = pytest.mark.smoke


def test_s03_cli_mock_roundtrip(tmp_path):
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
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "SUCCEEDED" and report["execution_starts"] == 1
    result = cli("inspect", "--workdir", str(tmp_path))
    assert result.returncode == 0
    assert json.loads(result.stdout)["actions"][0]["verified"]
    assert any(
        json.loads(line)["type"] == "verification"
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    )
