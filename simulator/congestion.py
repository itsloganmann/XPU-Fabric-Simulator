"""ECN marking and DCQCN congestion control."""

from simulator.gpu import Flow

# ECN marking threshold: mark packets when queue is at 50% capacity.
ECN_THRESHOLD = 0.5


def check_ecn(queue_depth: int, capacity: int) -> bool:
    """Return True if the queue utilization exceeds the ECN threshold."""
    if capacity <= 0:
        return False
    return (queue_depth / capacity) >= ECN_THRESHOLD


def dcqcn_adjust(flow: Flow, ecn_marked: bool):
    """Apply DCQCN rate control to a flow.

    On ECN: multiplicative decrease to back off quickly.
    No ECN: additive increase to slowly recover toward the base rate.
    """
    if ecn_marked:
        # Cut rate in half, but don't let it starve completely (min 5% of base_rate).
        flow.rate = max(flow.rate * 0.5, flow.base_rate * 0.05)
        flow.ecn_count += 1
    else:
        # Recover slowly by adding 5% of base rate back each tick.
        flow.rate = min(flow.rate + (flow.base_rate * 0.05), flow.base_rate)
