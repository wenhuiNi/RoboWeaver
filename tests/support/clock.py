"""A clock advanced by scenarios instead of wall-clock sleeps."""


class ManualClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Clock cannot move backwards")
        self.now += seconds
