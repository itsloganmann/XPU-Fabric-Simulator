"""Tests for ECMP and DLB routing algorithms."""

from simulator.routing import dlb_route, ecmp_route


def test_ecmp_deterministic():
    """ECMP should always return the same uplink for the same flow."""
    result1 = ecmp_route(flow_id=1, src_gpu=0, dst_gpu=5, num_uplinks=4)
    result2 = ecmp_route(flow_id=1, src_gpu=0, dst_gpu=5, num_uplinks=4)
    assert result1 == result2


def test_ecmp_range():
    """ECMP output should be within [0, num_uplinks)."""
    for flow_id in range(100):
        result = ecmp_route(flow_id=flow_id, src_gpu=0, dst_gpu=1, num_uplinks=4)
        assert 0 <= result < 4


def test_ecmp_distribution():
    """ECMP should distribute flows across uplinks (not all to one)."""
    uplinks_used = set()
    for flow_id in range(50):
        result = ecmp_route(flow_id=flow_id, src_gpu=0, dst_gpu=flow_id + 1, num_uplinks=4)
        uplinks_used.add(result)
    # With 50 different flows across 4 uplinks, we should hit at least 2.
    assert len(uplinks_used) >= 2


def test_dlb_picks_shortest_queue():
    """DLB should pick the uplink with the least queued packets."""
    queue_depths = {0: 10, 1: 5, 2: 20, 3: 3}
    assert dlb_route(queue_depths) == 3


def test_dlb_tiebreaker():
    """DLB returns a valid uplink even when queues are tied."""
    queue_depths = {0: 5, 1: 5, 2: 5}
    result = dlb_route(queue_depths)
    assert result in queue_depths


def test_dlb_single_uplink():
    """DLB works with a single uplink."""
    queue_depths = {0: 100}
    assert dlb_route(queue_depths) == 0
