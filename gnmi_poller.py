import time
import random
from prometheus_client import start_http_server, Gauge

# Prom metrics definitions
QUEUE_DEPTH = Gauge('switch_queue_depth_packets', 'Queue depth per port', ['switch', 'port', 'role'])
LINK_UTIL = Gauge('switch_link_utilization_pct', 'Link utilization percentage', ['switch', 'port', 'role'])
PACKETS_DROPPED = Gauge('switch_packets_dropped_total', 'Total packets dropped', ['switch'])
ECN_MARKS = Gauge('switch_ecn_marks_total', 'Total ECN marked packets', ['switch'])

def simulate_metrics():
    """Simulate realistic Arista switch telemetry for an AI Fabric."""
    roles = ['spine', 'leaf']
    num_spines = 4
    num_leaves = 8
    ports_per_switch = 16

    # Start Prom server
    start_http_server(8000)
    print("gNMI Poller Simulator listening on port 8000...")

    # Ongoing counters
    total_drops = {f"spine{i}": 0 for i in range(num_spines)}
    total_drops.update({f"leaf{i}": 0 for i in range(num_leaves)})
    
    total_ecn = {f"spine{i}": 0 for i in range(num_spines)}
    total_ecn.update({f"leaf{i}": 0 for i in range(num_leaves)})

    # Assume a slight imbalance on Spine 2 to simulate ECMP hashing issues intermittently
    while True:
        # Simulate load variations (sine wave + noise)
        base_load = 40 + (20 * random.random())
        is_ecmp_collision = random.random() > 0.7  # 30% chance of a burst hitting Spine 2

        # Spine telemetry
        for i in range(num_spines):
            sw_name = f"spine{i}"
            for p in range(ports_per_switch):
                if is_ecmp_collision and i == 2:
                    util = min(100.0, base_load + 40 + (10 * random.random()))
                    q_depth = int(util * 2.5)  # High queue
                    if util > 95:
                        total_drops[sw_name] += random.randint(1, 50)
                    if util > 80:
                        total_ecn[sw_name] += random.randint(10, 100)
                else:
                    util = min(100.0, base_load + (5 * random.random()))
                    q_depth = int(util * 0.5)

                QUEUE_DEPTH.labels(switch=sw_name, port=str(p), role="spine").set(q_depth)
                LINK_UTIL.labels(switch=sw_name, port=str(p), role="spine").set(util)

            PACKETS_DROPPED.labels(switch=sw_name).set(total_drops[sw_name])
            ECN_MARKS.labels(switch=sw_name).set(total_ecn[sw_name])

        # Leaf telemetry
        for i in range(num_leaves):
            sw_name = f"leaf{i}"
            for p in range(ports_per_switch):
                util = min(100.0, base_load + (15 * random.random()))
                q_depth = int(util * 0.8)
                if util > 80:
                    total_ecn[sw_name] += random.randint(5, 50)
                if util > 98:
                    total_drops[sw_name] += random.randint(0, 10)

                QUEUE_DEPTH.labels(switch=sw_name, port=str(p), role="leaf").set(q_depth)
                LINK_UTIL.labels(switch=sw_name, port=str(p), role="leaf").set(util)

            PACKETS_DROPPED.labels(switch=sw_name).set(total_drops[sw_name])
            ECN_MARKS.labels(switch=sw_name).set(total_ecn[sw_name])

        time.sleep(2)  # Emit metrics every 2s

if __name__ == '__main__':
    simulate_metrics()
