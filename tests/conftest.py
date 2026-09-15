"""Offline by default; live configuration never falls back to mock."""

import os

import pytest


@pytest.fixture(autouse=True)
def live_configuration(request):
    required = {
        "live_api": ["ROBOWEAVER_MODEL_CONFIG"],
        "live_sim": ["ROBOWEAVER_SIM_CONFIG"],
        "live_e2e": ["ROBOWEAVER_MODEL_CONFIG", "ROBOWEAVER_SIM_CONFIG"],
    }
    live = [name for name in required if request.node.get_closest_marker(name)]
    if live:
        missing = sorted({key for name in live for key in required[name] if not os.getenv(key)})
        if missing:
            pytest.fail(f"Missing live configuration: {', '.join(missing)}")
        request.getfixturevalue("socket_enabled")
