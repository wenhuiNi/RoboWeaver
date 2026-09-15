import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.smoke


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "roboweaver", *args], capture_output=True, text=True
    )


def test_s01_import_help_and_invalid_config(tmp_path):
    import roboweaver

    assert roboweaver.__version__
    result = cli("--help")
    assert result.returncode == 0 and "check" in result.stdout
    bad = tmp_path / "config.json"
    bad.write_text("{}")
    result = cli("check", "--config", str(bad))
    assert result.returncode != 0 and "mode" in result.stderr
    bad.write_text('{"mode":"mock"}')
    result = cli("check", "--config", str(bad))
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "valid"


def test_s01_network_is_disabled():
    import socket

    from pytest_socket import SocketBlockedError

    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
