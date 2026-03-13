"""Tests for the discrete-event simulation engine."""

from simulator.engine import EventLoop


def test_event_ordering():
    """Events should fire in chronological order."""
    engine = EventLoop()
    results = []

    engine.schedule(3.0, lambda _: results.append("third"))
    engine.schedule(1.0, lambda _: results.append("first"))
    engine.schedule(2.0, lambda _: results.append("second"))

    engine.run(until=5.0)
    assert results == ["first", "second", "third"]


def test_current_time():
    """Engine time should advance to the last processed event."""
    engine = EventLoop()
    engine.schedule(5.0, lambda _: None)
    engine.run(until=10.0)
    assert engine.now == 10.0


def test_events_beyond_until_not_fired():
    """Events scheduled beyond the 'until' time should not fire."""
    engine = EventLoop()
    results = []

    engine.schedule(1.0, lambda _: results.append("fired"))
    engine.schedule(10.0, lambda _: results.append("should_not_fire"))

    engine.run(until=5.0)
    assert results == ["fired"]
    assert not engine.is_empty()


def test_empty_engine():
    """An engine with no events should run without error."""
    engine = EventLoop()
    engine.run(until=100.0)
    assert engine.now == 100.0
    assert engine.is_empty()


def test_priority_ordering():
    """Events at the same time should fire in priority order."""
    engine = EventLoop()
    results = []

    engine.schedule(1.0, lambda _: results.append("low"), priority=10)
    engine.schedule(1.0, lambda _: results.append("high"), priority=1)

    engine.run(until=2.0)
    assert results == ["high", "low"]
