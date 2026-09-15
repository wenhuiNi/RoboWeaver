"""UNIX-socket client for a mock execution service owned by a separate process."""

import asyncio
import json

from roboweaver.contracts import ExecutionSnapshot, Observation


class RemoteExecutor:
    def __init__(self, path):
        self.path = str(path)

    async def call(self, operation, **payload):
        reader, writer = await asyncio.open_unix_connection(self.path)
        try:
            writer.write((json.dumps({"operation": operation, **payload}) + "\n").encode())
            await writer.drain()
            result = json.loads(await reader.readline())
            if "error" in result:
                raise ValueError(result["error"])
            return result["result"]
        finally:
            writer.close()
            await writer.wait_closed()

    async def submit(self, action):
        return ExecutionSnapshot.model_validate(
            await self.call("submit", action=action.model_dump(mode="json"))
        )

    async def lookup(self, idempotency_key):
        result = await self.call("lookup", key=idempotency_key)
        return ExecutionSnapshot.model_validate(result) if result else None

    async def cancel(self, idempotency_key):
        return ExecutionSnapshot.model_validate(await self.call("cancel", key=idempotency_key))

    async def observe(self):
        return Observation.model_validate(await self.call("observe"))
