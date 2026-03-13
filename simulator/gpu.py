"""GPU traffic generation and flow rate control."""

from dataclasses import dataclass, field


@dataclass
class Flow:
    """A traffic flow between two GPUs.

    Attributes:
        flow_id: Unique flow identifier.
        src_gpu: Source GPU index.
        dst_gpu: Destination GPU index.
        rate: Current sending rate in packets per time unit.
        base_rate: Maximum rate before congestion control.
        packets_sent: Total packets sent by this flow.
        packets_received: Total packets received.
        ecn_count: Number of ECN-marked packets received.
    """
    flow_id: int
    src_gpu: int
    dst_gpu: int
    rate: float = 10.0
    base_rate: float = 10.0
    packets_sent: int = 0
    packets_received: int = 0
    ecn_count: int = 0
    latencies: list[float] = field(default_factory=list)

    def record_latency(self, latency: float):
        self.latencies.append(latency)


@dataclass
class GPU:
    """A GPU node that generates and receives traffic flows.

    Attributes:
        gpu_id: Unique GPU identifier.
        leaf_id: The leaf switch this GPU is connected to.
        active_flows: Outbound flows keyed by flow_id.
    """
    gpu_id: int
    leaf_id: int
    active_flows: dict[int, Flow] = field(default_factory=dict)

    def create_flow(self, flow_id: int, dst_gpu: int, rate: float = 10.0) -> Flow:
        """Create a new outbound flow to a destination GPU."""
        flow = Flow(
            flow_id=flow_id,
            src_gpu=self.gpu_id,
            dst_gpu=dst_gpu,
            rate=rate,
            base_rate=rate,
        )
        self.active_flows[flow_id] = flow
        return flow
