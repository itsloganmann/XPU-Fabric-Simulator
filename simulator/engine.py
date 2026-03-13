"""Core discrete-event simulation engine using a priority-queue event loop."""

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class Event:
    """A scheduled event in the simulation timeline."""
    time: float
    priority: int = 0
    callback: Callable = field(compare=False, default=None)
    data: Any = field(compare=False, default=None)


class EventLoop:
    """Priority-queue driven event loop for discrete-event simulation."""

    def __init__(self):
        self._queue: list[Event] = []
        self._current_time: float = 0.0

    @property
    def now(self) -> float:
        return self._current_time

    def schedule(self, delay: float, callback: Callable, data: Any = None, priority: int = 0):
        """Schedule an event to fire after a delay from the current time."""
        event = Event(
            time=self._current_time + delay,
            priority=priority,
            callback=callback,
            data=data,
        )
        heapq.heappush(self._queue, event)
        return event

    def run(self, until: float):
        """Process events until the given simulation time."""
        while self._queue:
            event = self._queue[0]
            if event.time > until:
                break
            heapq.heappop(self._queue)
            self._current_time = event.time
            if event.callback:
                event.callback(event.data)
        self._current_time = until

    def is_empty(self) -> bool:
        return len(self._queue) == 0
