"""Metrics collection, time-series recording, and JSON export."""

import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MetricsCollector:
    """Collects per-tick simulation metrics for visualization and analysis.

    Records link utilization, queue depths, and flow latencies at each
    simulation tick for time-series charting and summary export.
    """

    # Time-series data: list of dicts, one per tick.
    ticks: list[dict] = field(default_factory=list)

    # Per-flow latency samples for percentile calculations.
    flow_latencies: dict[int, list[float]] = field(default_factory=dict)

    # Summary counters accumulated across the simulation.
    total_packets_sent: int = 0
    total_packets_dropped: int = 0
    total_ecn_marks: int = 0

    def record_tick(self, tick: int, switch_data: list[dict], link_utils: list[dict]):
        """Record metrics for a single simulation tick.

        Args:
            tick: Current simulation time step.
            switch_data: List of dicts with switch_name, port, queue_depth, utilization.
            link_utils: List of dicts with link_name, utilization.
        """
        for entry in switch_data:
            self.ticks.append({"tick": tick, "type": "queue", **entry})
        for entry in link_utils:
            self.ticks.append({"tick": tick, "type": "link", **entry})

    def record_latency(self, flow_id: int, latency: float):
        """Record a latency sample for a flow."""
        self.flow_latencies.setdefault(flow_id, [])
        self.flow_latencies[flow_id].append(latency)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time-series data to a pandas DataFrame for charting."""
        return pd.DataFrame(self.ticks)

    def latency_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame of per-tick p99 latency across all flows."""
        if not self.flow_latencies:
            return pd.DataFrame(columns=["tick", "p99_latency"])

        # Aggregate all latency samples into tick-based buckets.
        all_samples = []
        for flow_id, lats in self.flow_latencies.items():
            for i, lat in enumerate(lats):
                all_samples.append({"tick": i, "latency": lat})

        if not all_samples:
            return pd.DataFrame(columns=["tick", "p99_latency"])

        df = pd.DataFrame(all_samples)
        p99 = df.groupby("tick")["latency"].quantile(0.99).reset_index()
        p99.columns = ["tick", "p99_latency"]
        return p99

    def summary_json(self, switches) -> str:
        """Export a JSON summary of the simulation for LLM analysis.

        Args:
            switches: List of Switch objects from the simulation.

        Returns:
            JSON string with per-switch metrics and overall statistics.
        """
        summary = {
            "total_packets_sent": self.total_packets_sent,
            "total_packets_dropped": self.total_packets_dropped,
            "total_ecn_marks": self.total_ecn_marks,
            "drop_rate_pct": round(
                (self.total_packets_dropped / max(self.total_packets_sent, 1)) * 100, 2
            ),
            "switches": {},
        }

        for sw in switches:
            summary["switches"][sw.name] = {
                "total_drops": sw.total_drops,
                "max_queue_utilization_pct": round(sw.max_queue_utilization * 100, 1),
                "per_port_drops": dict(sw.drops),
                "per_port_peak_queue": dict(sw.peak_queue),
            }

        # Add latency percentiles.
        all_lats = []
        for lats in self.flow_latencies.values():
            all_lats.extend(lats)

        if all_lats:
            all_lats.sort()
            n = len(all_lats)
            summary["latency"] = {
                "p50": round(all_lats[int(n * 0.5)], 2),
                "p95": round(all_lats[int(n * 0.95)], 2),
                "p99": round(all_lats[int(n * 0.99)], 2),
                "max": round(all_lats[-1], 2),
            }

        return json.dumps(summary, indent=2)
