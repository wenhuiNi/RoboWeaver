"""Command-line entry points; operational errors have nonzero exit codes."""

import argparse
import json
from pathlib import Path

from roboweaver import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roboweaver", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="List declared action groups (not execution availability)")
    check = commands.add_parser("check", help="Validate offline runtime configuration")
    check.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        from roboweaver.catalog import ACTION_GROUPS

        print(json.dumps({"declared_groups": ACTION_GROUPS, "execution_verified": False}))
        return 0
    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or config.get("mode") != "mock":
            raise ValueError("mode must be 'mock'; live adapters are not configured")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "valid", "mode": "mock"}))
    return 0
