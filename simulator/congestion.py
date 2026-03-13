"""ECN marking and DCQCN congestion control."""

from simulator.gpu import Flow

# ECN marking threshold: mark packets when queue is at 80% capacity.
ECN_THRESHOLD = 0.8

# DCQCN rate adjustment parameters.
DCQCN_DECREASE_FACTOR = 0.5  # Multiplicative decrease on congestion.
DCQCN_INCREASE_STEP = 0.5    # Additive increase during recovery.
DCQCN_MIN_RATE = 1.0         # Floor to prevent starvation.


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
        flow.rate = max(flow.rate * DCQCN_DECREASE_FACTOR, DCQCN_MIN_RATE)
        flow.ecn_count += 1
    else:
        flow.rate = min(flow.rate + DCQCN_INCREASE_STEP, flow.base_rate)
