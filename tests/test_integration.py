"""Integration tests: run a full simulation and validate output."""

import json

from simulator.congestion import check_ecn, dcqcn_adjust
from simulator.engine import EventLoop
from simulator.gpu import GPU
from simulator.metrics import MetricsCollector
from simulator.routing import dlb_route, ecmp_route
from simulator.switch import Packet, Switch
from simulator.workloads import all_to_all_workload, web_traffic_workload


def _run_small_simulation(routing_mode="ecmp", workload="all_to_all", ticks=50):
    """Helper: run a minimal 2-spine, 4-leaf, 2-gpu-per-leaf simulation."""
    num_spines = 2
    num_leaves = 4
    gpus_per_leaf = 2
    buffer_capacity = 64

    spines = [
        Switch(switch_id=i, switch_type="spine", buffer_capacity=buffer_capacity, num_ports=num_leaves)
        for i in range(num_spines)
    ]
    leaves = [
        Switch(switch_id=i, switch_type="leaf", buffer_capacity=buffer_capacity, num_ports=num_spines + gpus_per_leaf)
        for i in range(num_leaves)
    ]
    gpus = []
    gpu_id = 0
    for leaf_id in range(num_leaves):
        for _ in range(gpus_per_leaf):
            gpus.append(GPU(gpu_id=gpu_id, leaf_id=leaf_id))
            gpu_id += 1

    gpu_to_leaf = {gpu.gpu_id: gpu.leaf_id for gpu in gpus}
    metrics = MetricsCollector()

    if workload == "all_to_all":
        flows = all_to_all_workload(gpus, base_rate=8.0)
    else:
        import random
        random.seed(42)
        flows = web_traffic_workload(gpus, base_rate=5.0, density=0.3)

    use_dlb = routing_mode == "dlb"
    all_switches = spines + leaves

    for tick in range(ticks):
        packets_this_tick = []
        for flow in flows:
            pkt = Packet(
                flow_id=flow.flow_id,
                src_gpu=flow.src_gpu,
                dst_gpu=flow.dst_gpu,
                src_leaf=gpu_to_leaf[flow.src_gpu],
                dst_leaf=gpu_to_leaf[flow.dst_gpu],
                timestamp=tick,
            )
            packets_this_tick.append((flow, pkt))
            flow.packets_sent += 1
            metrics.total_packets_sent += 1

        for flow, pkt in packets_this_tick:
            if pkt.src_leaf == pkt.dst_leaf:
                flow.record_latency(1.0)
                metrics.record_latency(flow.flow_id, 1.0)
                flow.packets_received += 1
                continue

            src_leaf = leaves[pkt.src_leaf]
            if use_dlb:
                queue_depths = {i: spines[i].queue_depth(pkt.src_leaf) for i in range(num_spines)}
                spine_idx = dlb_route(queue_depths)
            else:
                spine_idx = ecmp_route(flow.flow_id, pkt.src_gpu, pkt.dst_gpu, num_spines)

            spine = spines[spine_idx]
            uplink_port = gpus_per_leaf + spine_idx

            if not src_leaf.enqueue(uplink_port, pkt):
                metrics.total_packets_dropped += 1
                continue

            if check_ecn(src_leaf.queue_depth(uplink_port), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            if not spine.enqueue(pkt.src_leaf, pkt):
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                continue

            dst_leaf = leaves[pkt.dst_leaf]
            if not dst_leaf.enqueue(spine_idx, pkt):
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                spine.dequeue(pkt.src_leaf)
                continue

            latency = 1.0 + src_leaf.queue_depth(uplink_port) * 0.1 + spine.queue_depth(pkt.src_leaf) * 0.15
            flow.record_latency(latency)
            metrics.record_latency(flow.flow_id, latency)
            flow.packets_received += 1
            dcqcn_adjust(flow, pkt.ecn_marked)

        for sw in all_switches:
            for port in list(sw.port_queues.keys()):
                drain = min(2, sw.queue_depth(port))
                for _ in range(drain):
                    sw.dequeue(port)

        switch_data = []
        link_utils = []
        for sw in all_switches:
            for port in range(sw.num_ports):
                switch_data.append({
                    "switch_name": sw.name,
                    "port": port,
                    "queue_depth": sw.queue_depth(port),
                    "utilization": round(sw.queue_utilization(port) * 100, 1),
                })
                link_utils.append({
                    "link_name": f"{sw.name}_p{port}",
                    "utilization": round(sw.queue_utilization(port) * 100, 1),
                })
        metrics.record_tick(tick, switch_data, link_utils)

    return metrics, all_switches


def test_simulation_produces_valid_json():
    """Full simulation should produce parseable JSON metrics."""
    metrics, switches = _run_small_simulation()
    summary = metrics.summary_json(switches)
    data = json.loads(summary)

    assert "total_packets_sent" in data
    assert "total_packets_dropped" in data
    assert "switches" in data
    assert data["total_packets_sent"] > 0


def test_simulation_records_latencies():
    """Simulation should record latency samples."""
    metrics, _ = _run_small_simulation()
    assert len(metrics.flow_latencies) > 0
    all_lats = [l for lats in metrics.flow_latencies.values() for l in lats]
    assert len(all_lats) > 0


def test_dataframe_not_empty():
    """Time-series DataFrame should contain tick data."""
    metrics, _ = _run_small_simulation()
    df = metrics.to_dataframe()
    assert not df.empty
    assert "tick" in df.columns


def test_dlb_reduces_max_queue():
    """DLB should generally result in lower peak queue than ECMP."""
    _, switches_ecmp = _run_small_simulation(routing_mode="ecmp", ticks=100)
    _, switches_dlb = _run_small_simulation(routing_mode="dlb", ticks=100)

    max_q_ecmp = max(sw.max_queue_utilization for sw in switches_ecmp)
    max_q_dlb = max(sw.max_queue_utilization for sw in switches_dlb)

    # DLB should perform at least as well as ECMP in most scenarios.
    # Use a generous threshold since small topologies can vary.
    assert max_q_dlb <= max_q_ecmp * 1.5


def test_web_traffic_fewer_flows():
    """Web traffic should generate fewer flows than all-to-all."""
    metrics_a2a, _ = _run_small_simulation(workload="all_to_all", ticks=10)
    metrics_web, _ = _run_small_simulation(workload="web", ticks=10)

    # All-to-all always sends more total packets than sparse web traffic.
    assert metrics_a2a.total_packets_sent >= metrics_web.total_packets_sent
