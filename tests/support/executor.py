"""Event injection foundation; execution behavior is added with the scheduler."""

import asyncio


class EventSource:
    def __init__(self):
        self.events = asyncio.Queue()

    def emit(self, event):
        self.events.put_nowait(event)

    async def receive(self):
        return await self.events.get()
