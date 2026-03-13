"""Tests for ECN marking and DCQCN congestion control."""

from simulator.congestion import (
    DCQCN_DECREASE_FACTOR,
    DCQCN_INCREASE_STEP,
    DCQCN_MIN_RATE,
    ECN_THRESHOLD,
    check_ecn,
    dcqcn_adjust,
)
from simulator.gpu import Flow


def test_ecn_below_threshold():
    """No ECN mark when queue is under 80%."""
    assert not check_ecn(queue_depth=70, capacity=100)


def test_ecn_at_threshold():
    """ECN marks at exactly 80% utilization."""
    assert check_ecn(queue_depth=80, capacity=100)


def test_ecn_above_threshold():
    """ECN marks above 80% utilization."""
    assert check_ecn(queue_depth=95, capacity=100)


def test_ecn_zero_capacity():
    """No ECN mark when capacity is zero (edge case)."""
    assert not check_ecn(queue_depth=0, capacity=0)


def test_dcqcn_decrease_on_ecn():
    """Rate should decrease multiplicatively on ECN."""
    flow = Flow(flow_id=0, src_gpu=0, dst_gpu=1, rate=10.0, base_rate=10.0)
    dcqcn_adjust(flow, ecn_marked=True)
    assert flow.rate == 10.0 * DCQCN_DECREASE_FACTOR
    assert flow.ecn_count == 1


def test_dcqcn_increase_without_ecn():
    """Rate should increase additively without ECN, capped at base rate."""
    flow = Flow(flow_id=0, src_gpu=0, dst_gpu=1, rate=5.0, base_rate=10.0)
    dcqcn_adjust(flow, ecn_marked=False)
    assert flow.rate == 5.0 + DCQCN_INCREASE_STEP


def test_dcqcn_no_exceed_base_rate():
    """Rate should not exceed the base rate during recovery."""
    flow = Flow(flow_id=0, src_gpu=0, dst_gpu=1, rate=9.8, base_rate=10.0)
    dcqcn_adjust(flow, ecn_marked=False)
    assert flow.rate == 10.0


def test_dcqcn_min_rate_floor():
    """Rate should not drop below the minimum floor."""
    flow = Flow(flow_id=0, src_gpu=0, dst_gpu=1, rate=DCQCN_MIN_RATE, base_rate=10.0)
    dcqcn_adjust(flow, ecn_marked=True)
    assert flow.rate == DCQCN_MIN_RATE


def test_dcqcn_repeated_ecn():
    """Multiple ECN events should converge the rate toward the floor."""
    flow = Flow(flow_id=0, src_gpu=0, dst_gpu=1, rate=10.0, base_rate=10.0)
    for _ in range(20):
        dcqcn_adjust(flow, ecn_marked=True)
    assert flow.rate == DCQCN_MIN_RATE
