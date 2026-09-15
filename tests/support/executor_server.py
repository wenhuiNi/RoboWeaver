"""Independent synthetic execution service used in process-crash acceptance tests."""

import argparse
import asyncio
import json
from pathlib import Path

from roboweaver.adapters.mock import MockExecutor
from roboweaver.contracts import ActionRecord, ActionStatus


async def serve(path, database):
    executor = MockExecutor(database)

    async def handle(reader, writer):
        try:
            request = json.loads(await reader.readline())
            operation = request["operation"]
            if operation == "submit":
                result = await executor.submit(ActionRecord.model_validate(request["action"]))
            elif operation == "lookup":
                result = await executor.lookup(request["key"])
            elif operation == "cancel":
                result = await executor.cancel(request["key"])
            elif operation == "observe":
                result = await executor.observe()
            elif operation == "inject":
                result = executor.inject(
                    request["key"],
                    ActionStatus(request.get("status", "COMPLETED")),
                    state=request.get("state"),
                )
            elif operation == "starts":
                result = executor.starts
            else:
                raise ValueError("Unknown operation")
            response = {
                "result": result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else result
            }
        except Exception as exc:
            response = {"error": str(exc)}
        writer.write((json.dumps(response) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=str(path))
    print("READY", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.socket, args.database))
