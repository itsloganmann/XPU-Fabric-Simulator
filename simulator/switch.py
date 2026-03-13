"""Switch models for Leaf and Spine network elements."""

from dataclasses import dataclass, field


@dataclass
class Packet:
    """Represents a network packet traversing the fabric."""
    flow_id: int
    src_gpu: int
    dst_gpu: int
    src_leaf: int
    dst_leaf: int
    size: int = 1500  # bytes
    ecn_marked: bool = False
    timestamp: float = 0.0


@dataclass
class Switch:
    """A network switch (Leaf or Spine) with per-port queues.

    Attributes:
        switch_id: Unique identifier for this switch.
        switch_type: Either 'leaf' or 'spine'.
        buffer_capacity: Maximum packets per port queue.
        num_ports: Number of ports on this switch.
    """
    switch_id: int
    switch_type: str  # 'leaf' or 'spine'
    buffer_capacity: int = 128
    num_ports: int = 8
    port_queues: dict[int, list[Packet]] = field(default_factory=dict)
    drops: dict[int, int] = field(default_factory=dict)
    peak_queue: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for port in range(self.num_ports):
            self.port_queues.setdefault(port, [])
            self.drops.setdefault(port, 0)
            self.peak_queue.setdefault(port, 0)

    def enqueue(self, port: int, packet: Packet) -> bool:
        """Add a packet to a port queue. Returns False if the queue is full (drop)."""
        if port not in self.port_queues:
            self.port_queues[port] = []
            self.drops[port] = 0
            self.peak_queue[port] = 0

        queue = self.port_queues[port]
        if len(queue) >= self.buffer_capacity:
            self.drops[port] += 1
            return False

        queue.append(packet)

        # Track peak utilization
        current_depth = len(queue)
        if current_depth > self.peak_queue.get(port, 0):
            self.peak_queue[port] = current_depth

        return True

    def dequeue(self, port: int) -> Packet | None:
        """Remove and return the next packet from a port queue."""
        queue = self.port_queues.get(port, [])
        if queue:
            return queue.pop(0)
        return None

    def queue_depth(self, port: int) -> int:
        """Current number of packets in a port queue."""
        return len(self.port_queues.get(port, []))

    def queue_utilization(self, port: int) -> float:
        """Current queue utilization as a fraction [0.0, 1.0]."""
        return self.queue_depth(port) / self.buffer_capacity

    @property
    def total_drops(self) -> int:
        return sum(self.drops.values())

    @property
    def max_queue_utilization(self) -> float:
        """Highest queue utilization across all ports."""
        if not self.peak_queue:
            return 0.0
        return max(self.peak_queue.values()) / self.buffer_capacity

    @property
    def name(self) -> str:
        return f"{self.switch_type}_{self.switch_id}"
