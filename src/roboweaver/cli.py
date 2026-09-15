"""Command-line entry points; operational errors have nonzero exit codes."""

import argparse
import asyncio
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
    run = commands.add_parser("run", help="Run an explicitly selected synthetic scenario")
    run.add_argument("--mode", choices=["mock"], required=True)
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--capabilities", type=Path, required=True)
    run.add_argument("--scenario", type=Path, required=True)
    run.add_argument("--workdir", type=Path, required=True)
    run.add_argument("--until-waiting", action="store_true")
    inspect = commands.add_parser(
        "inspect", help="Read persisted run state without executing actions"
    )
    inspect.add_argument("--workdir", type=Path, required=True)
    cancel = commands.add_parser("cancel", help="Request stop and report confirmed execution state")
    cancel.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        from roboweaver.catalog import ACTION_GROUPS

        print(json.dumps({"declared_groups": ACTION_GROUPS, "execution_verified": False}))
        return 0
    if args.command in {"run", "inspect", "cancel"}:
        try:
            if args.command == "cancel":
                from roboweaver.offline import cancel_mock

                result = asyncio.run(cancel_mock(args.workdir))
                print(json.dumps(result))
                return 0 if result["status"] in {"CANCELED", "SUCCEEDED", "FAILED"} else 1
            if args.command == "run":
                from roboweaver.offline import run_mock

                result = asyncio.run(
                    run_mock(
                        args.task,
                        args.capabilities,
                        args.scenario,
                        args.workdir,
                        args.until_waiting,
                    )
                )
                print(json.dumps(result))
                return 0 if result["status"] == "SUCCEEDED" else 1
            from roboweaver.ledger import Ledger

            meta = json.loads((args.workdir / "run.json").read_text())
            print(Ledger(args.workdir / "ledger.db").read(meta["run_id"]).model_dump_json())
            return 0
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            parser.error(str(exc))
    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or config.get("mode") != "mock":
            raise ValueError("mode must be 'mock'; live adapters are not configured")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "valid", "mode": "mock"}))
    return 0
