"""Workload generators for different traffic patterns."""

from simulator.gpu import GPU, Flow


def all_to_all_workload(gpus: list[GPU], base_rate: float = 10.0) -> list[Flow]:
    """Generate an all-to-all traffic pattern (AI training collective).

    Every GPU sends a flow to every other GPU, simulating the communication
    pattern of distributed model training (e.g., all-reduce).
    """
    flows = []
    flow_id = 0
    for src in gpus:
        for dst in gpus:
            if src.gpu_id != dst.gpu_id:
                flow = src.create_flow(flow_id, dst.gpu_id, rate=base_rate)
                flows.append(flow)
                flow_id += 1
    return flows


def web_traffic_workload(gpus: list[GPU], base_rate: float = 5.0, density: float = 0.2) -> list[Flow]:
    """Generate sparse random traffic (web/storage pattern).

    Only a fraction of GPU pairs communicate, simulating typical
    request-response or client-server traffic patterns.
    """
    import random
    flows = []
    flow_id = 0
    for src in gpus:
        for dst in gpus:
            if src.gpu_id != dst.gpu_id and random.random() < density:
                flow = src.create_flow(flow_id, dst.gpu_id, rate=base_rate)
                flows.append(flow)
                flow_id += 1
    return flows
