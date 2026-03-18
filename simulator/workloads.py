"""Workload generators for different traffic patterns."""

from simulator.gpu import GPU, Flow


def all_to_all_workload(gpus: list[GPU], base_rate: float = 4.0, mtu: int = 1500) -> list[Flow]:
    """Generate an all-to-all traffic pattern (AI training collective).

    Every GPU sends a flow to every other GPU, simulating the communication
    pattern of distributed model training (e.g., all-reduce).
    """
    flows = []
    flow_id = 0
    num_peers = len(gpus) - 1
    if num_peers <= 0:
        return flows
        
    # Scale raw packet count by MTU (9000 bytes yields ~6x fewer packets)
    mtu_factor = 1500.0 / mtu
    per_flow_rate = (base_rate / num_peers) * mtu_factor
    for src in gpus:
        for dst in gpus:
            if src.gpu_id != dst.gpu_id:
                flow = src.create_flow(flow_id, dst.gpu_id, rate=per_flow_rate)
                flows.append(flow)
                flow_id += 1
    return flows


def web_traffic_workload(gpus: list[GPU], base_rate: float = 2.0, density: float = 0.2, mtu: int = 1500) -> list[Flow]:
    """Generate sparse random traffic (web/storage pattern).

    Only a fraction of GPU pairs communicate, simulating typical
    request-response or client-server traffic patterns.
    """
    import random
    flows = []
    flow_id = 0
    mtu_factor = 1500.0 / mtu
    
    # Pre-calculate active flows per source to distribute base_rate evenly
    for src in gpus:
        active_dsts = [dst for dst in gpus if src.gpu_id != dst.gpu_id and random.random() < density]
        if not active_dsts:
            continue
            
        per_flow_rate = (base_rate / len(active_dsts)) * mtu_factor
        for dst in active_dsts:
            flow = src.create_flow(flow_id, dst.gpu_id, rate=per_flow_rate)
            flows.append(flow)
            flow_id += 1
            
    return flows
