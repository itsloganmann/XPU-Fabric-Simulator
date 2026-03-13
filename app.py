"""Discrete-Event Network Simulator -- Streamlit web application.

Interactive data center network simulator with configurable Leaf-Spine
topology, ECMP/DLB routing, ECN/DCQCN congestion control, and LLM analysis.
"""

import random
import time

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.llm_agent import analyze_metrics
from simulator.congestion import check_ecn, dcqcn_adjust
from simulator.engine import EventLoop
from simulator.gpu import GPU, Flow
from simulator.metrics import MetricsCollector
from simulator.routing import dlb_route, ecmp_route
from simulator.switch import Packet, Switch
from simulator.workloads import all_to_all_workload, web_traffic_workload

# -- Page configuration --

st.set_page_config(
    page_title="Data Center Network Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for a polished look --

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #0088ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #8899aa;
        font-size: 1.0rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3d);
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8899aa;
        margin-top: 0.3rem;
    }
    .analysis-box {
        background: linear-gradient(135deg, #1a1f2e, #1e2536);
        border: 1px solid #2a3545;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117, #151b28);
    }
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -- Sidebar controls --

with st.sidebar:
    st.markdown("### Network Topology")
    num_spines = st.select_slider(
        "Spine Switches",
        options=[2, 4, 8],
        value=4,
        help="Number of spine switches in the fabric.",
    )
    num_leaves = st.select_slider(
        "Leaf Switches",
        options=[4, 8, 16],
        value=8,
        help="Number of leaf (ToR) switches.",
    )
    gpus_per_leaf = st.select_slider(
        "GPUs per Leaf",
        options=[4, 8, 16],
        value=4,
        help="Number of GPU endpoints per leaf switch.",
    )

    st.markdown("---")
    st.markdown("### Simulation Settings")
    routing_mode = st.selectbox(
        "Routing Mode",
        ["ECMP", "Dynamic Load Balancing"],
        help="ECMP uses hash-based path selection. DLB uses real-time queue depths.",
    )
    workload_type = st.selectbox(
        "Workload Type",
        ["All-to-All AI Training", "Web Traffic"],
        help="All-to-All simulates distributed training. Web Traffic is sparse and random.",
    )
    sim_duration = st.slider(
        "Simulation Ticks",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of time steps to simulate.",
    )
    buffer_capacity = st.slider(
        "Buffer Capacity (packets)",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
        help="Per-port buffer depth on each switch.",
    )

    st.markdown("---")
    run_button = st.button("Run Simulation", type="primary", width="stretch")


# -- Header --

st.markdown('<div class="main-header">Data Center Network Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Discrete-event simulation of Leaf-Spine fabrics with ECMP, DLB, and ECN/DCQCN congestion control"
    "</div>",
    unsafe_allow_html=True,
)


def build_topology(num_spines, num_leaves, gpus_per_leaf, buffer_capacity):
    """Construct the Leaf-Spine topology with GPU endpoints."""
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

    # Map GPU IDs to their leaf switch index.
    gpu_to_leaf = {gpu.gpu_id: gpu.leaf_id for gpu in gpus}

    return spines, leaves, gpus, gpu_to_leaf


def run_simulation(num_spines, num_leaves, gpus_per_leaf, routing_mode,
                   workload_type, sim_duration, buffer_capacity):
    """Execute the discrete-event simulation and return collected metrics."""
    spines, leaves, gpus, gpu_to_leaf = build_topology(
        num_spines, num_leaves, gpus_per_leaf, buffer_capacity
    )
    metrics = MetricsCollector()
    engine = EventLoop()

    # Seed for reproducibility when using web traffic.
    random.seed(42)

    # Generate workload flows.
    base_rate = 10.0 if workload_type == "All-to-All AI Training" else 5.0
    if workload_type == "All-to-All AI Training":
        flows = all_to_all_workload(gpus, base_rate=base_rate)
    else:
        flows = web_traffic_workload(gpus, base_rate=base_rate, density=0.15)

    if not flows:
        return None, None, None, "No flows generated. Try a different configuration."

    all_switches = spines + leaves
    use_dlb = routing_mode == "Dynamic Load Balancing"

    # Run the simulation tick by tick.
    for tick in range(sim_duration):
        packets_this_tick = []

        # Each flow generates packets based on its current rate.
        for flow in flows:
            num_packets = max(1, int(flow.rate / len(flows) * 2 + 0.5))
            # Scale packet generation to keep simulation tractable.
            num_packets = min(num_packets, 3)

            for _ in range(num_packets):
                src_leaf_id = gpu_to_leaf[flow.src_gpu]
                dst_leaf_id = gpu_to_leaf[flow.dst_gpu]

                pkt = Packet(
                    flow_id=flow.flow_id,
                    src_gpu=flow.src_gpu,
                    dst_gpu=flow.dst_gpu,
                    src_leaf=src_leaf_id,
                    dst_leaf=dst_leaf_id,
                    timestamp=tick,
                )
                packets_this_tick.append((flow, pkt))
                flow.packets_sent += 1
                metrics.total_packets_sent += 1

        # Route each packet through the fabric.
        for flow, pkt in packets_this_tick:
            src_leaf = leaves[pkt.src_leaf]
            dst_leaf = leaves[pkt.dst_leaf]

            # Intra-leaf traffic skips the spine layer.
            if pkt.src_leaf == pkt.dst_leaf:
                # Deliver directly within the same leaf.
                latency = 1.0 + src_leaf.queue_depth(0) * 0.1
                flow.record_latency(latency)
                metrics.record_latency(flow.flow_id, latency)
                flow.packets_received += 1
                continue

            # Select a spine uplink.
            if use_dlb:
                # Query real-time queue depths on each spine's port for this leaf.
                queue_depths = {}
                for spine_idx, spine in enumerate(spines):
                    queue_depths[spine_idx] = spine.queue_depth(pkt.src_leaf)
                spine_idx = dlb_route(queue_depths)
            else:
                spine_idx = ecmp_route(
                    flow.flow_id, pkt.src_gpu, pkt.dst_gpu, num_spines
                )

            spine = spines[spine_idx]

            # Enqueue on the source leaf's uplink port.
            uplink_port = gpus_per_leaf + spine_idx
            if not src_leaf.enqueue(uplink_port, pkt):
                metrics.total_packets_dropped += 1
                continue

            # Check ECN at source leaf.
            if check_ecn(src_leaf.queue_depth(uplink_port), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the spine's port for the source leaf.
            spine_port_in = pkt.src_leaf
            if not spine.enqueue(spine_port_in, pkt):
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                continue

            # Check ECN at spine.
            if check_ecn(spine.queue_depth(spine_port_in), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the spine's output port toward the destination leaf.
            spine_port_out = pkt.dst_leaf
            if not spine.enqueue(spine_port_out, pkt):
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                spine.dequeue(spine_port_in)
                continue

            # Check ECN at spine output.
            if check_ecn(spine.queue_depth(spine_port_out), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the destination leaf's downlink.
            downlink_port = spine_idx
            if not dst_leaf.enqueue(downlink_port, pkt):
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                spine.dequeue(spine_port_in)
                spine.dequeue(spine_port_out)
                continue

            # Packet delivered. Calculate latency from queuing delays.
            latency = (
                1.0
                + src_leaf.queue_depth(uplink_port) * 0.1
                + spine.queue_depth(spine_port_in) * 0.15
                + spine.queue_depth(spine_port_out) * 0.15
                + dst_leaf.queue_depth(downlink_port) * 0.1
            )
            flow.record_latency(latency)
            metrics.record_latency(flow.flow_id, latency)
            flow.packets_received += 1

            # Apply DCQCN rate adjustment based on ECN feedback.
            dcqcn_adjust(flow, pkt.ecn_marked)

        # Drain queues partially each tick (simulate forwarding).
        for sw in all_switches:
            for port in list(sw.port_queues.keys()):
                drain_count = min(3, sw.queue_depth(port))
                for _ in range(drain_count):
                    sw.dequeue(port)

        # Record per-tick metrics.
        switch_data = []
        link_utils = []
        for sw in all_switches:
            for port in range(sw.num_ports):
                depth = sw.queue_depth(port)
                util = sw.queue_utilization(port)
                switch_data.append({
                    "switch_name": sw.name,
                    "port": port,
                    "queue_depth": depth,
                    "utilization": round(util * 100, 1),
                })
                link_utils.append({
                    "link_name": f"{sw.name}_p{port}",
                    "utilization": round(util * 100, 1),
                })

        metrics.record_tick(tick, switch_data, link_utils)

    summary_json = metrics.summary_json(all_switches)
    return metrics, all_switches, summary_json, None


def create_charts(metrics, all_switches):
    """Build Plotly charts from simulation metrics."""
    df = metrics.to_dataframe()

    chart_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(size=10),
        ),
    )

    charts = {}

    # 1. Link utilization over time (aggregate by switch).
    link_df = df[df["type"] == "link"].copy()
    if not link_df.empty:
        # Extract switch name from link_name for aggregation.
        link_df["switch"] = link_df["link_name"].apply(lambda x: "_".join(x.split("_")[:-1]))
        agg = link_df.groupby(["tick", "switch"])["utilization"].mean().reset_index()

        fig_link = px.line(
            agg,
            x="tick",
            y="utilization",
            color="switch",
            title="Link Utilization Over Time",
            labels={"tick": "Time (ticks)", "utilization": "Utilization (%)", "switch": "Switch"},
        )
        fig_link.update_layout(**chart_layout)
        fig_link.update_traces(line=dict(width=2))
        charts["link_util"] = fig_link

    # 2. Queue depths over time (top switches by peak depth).
    queue_df = df[df["type"] == "queue"].copy()
    if not queue_df.empty:
        agg_q = queue_df.groupby(["tick", "switch_name"])["queue_depth"].max().reset_index()
        # Pick the top 8 most congested switches for readability.
        top_switches = (
            agg_q.groupby("switch_name")["queue_depth"]
            .max()
            .nlargest(8)
            .index.tolist()
        )
        agg_q_filtered = agg_q[agg_q["switch_name"].isin(top_switches)]

        fig_queue = px.line(
            agg_q_filtered,
            x="tick",
            y="queue_depth",
            color="switch_name",
            title="Queue Depth Over Time (Top Switches)",
            labels={"tick": "Time (ticks)", "queue_depth": "Queue Depth (packets)", "switch_name": "Switch"},
        )
        fig_queue.update_layout(**chart_layout)
        fig_queue.update_traces(line=dict(width=2))
        charts["queue_depth"] = fig_queue

    # 3. Tail latency (p99) over time.
    lat_df = metrics.latency_dataframe()
    if not lat_df.empty:
        fig_lat = px.line(
            lat_df,
            x="tick",
            y="p99_latency",
            title="Tail Latency (p99) Over Time",
            labels={"tick": "Time (ticks)", "p99_latency": "p99 Latency"},
        )
        fig_lat.update_layout(**chart_layout)
        fig_lat.update_traces(
            line=dict(width=2.5, color="#ff6b6b"),
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.1)",
        )
        charts["latency"] = fig_lat

    return charts


# -- Main panel --

if run_button:
    with st.spinner("Running simulation..."):
        start_time = time.time()
        metrics, switches, summary_json, error = run_simulation(
            num_spines=num_spines,
            num_leaves=num_leaves,
            gpus_per_leaf=gpus_per_leaf,
            routing_mode=routing_mode,
            workload_type=workload_type,
            sim_duration=sim_duration,
            buffer_capacity=buffer_capacity,
        )
        elapsed = time.time() - start_time

    if error:
        st.error(error)
    else:
        # Summary metrics row.
        import json
        summary = json.loads(summary_json)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{summary["total_packets_sent"]:,}</div>'
                f'<div class="metric-label">Packets Sent</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{summary["total_packets_dropped"]:,}</div>'
                f'<div class="metric-label">Packets Dropped</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{summary["drop_rate_pct"]}%</div>'
                f'<div class="metric-label">Drop Rate</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            latency = summary.get("latency", {})
            p99_val = latency.get("p99", "N/A")
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{p99_val}</div>'
                f'<div class="metric-label">p99 Latency</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"*Simulation completed in {elapsed:.2f}s*")
        st.markdown("---")

        # Charts.
        charts = create_charts(metrics, switches)

        if "link_util" in charts:
            st.plotly_chart(charts["link_util"], width="stretch")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if "queue_depth" in charts:
                st.plotly_chart(charts["queue_depth"], width="stretch")
        with chart_col2:
            if "latency" in charts:
                st.plotly_chart(charts["latency"], width="stretch")

        st.markdown("---")

        # AI analysis panel.
        st.markdown("### NetOps AI Analysis")
        with st.spinner("Analyzing metrics..."):
            analysis = analyze_metrics(summary_json, routing_mode)
        st.markdown(
            f'<div class="analysis-box">{analysis}</div>',
            unsafe_allow_html=True,
        )

        # Expandable raw metrics.
        with st.expander("Raw Simulation Metrics (JSON)"):
            st.code(summary_json, language="json")

else:
    # Default landing state.
    st.info("Configure the network topology in the sidebar and click **Run Simulation** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "#### Leaf-Spine Fabric\n"
            "Simulate data center networks with configurable "
            "spine and leaf switch counts, buffer depths, and GPU endpoints."
        )
    with col2:
        st.markdown(
            "#### ECMP vs DLB\n"
            "Compare hash-based Equal-Cost Multi-Path routing against "
            "Dynamic Load Balancing that uses real-time queue feedback."
        )
    with col3:
        st.markdown(
            "#### ECN/DCQCN\n"
            "Watch congestion control in action: ECN marks packets at 80% "
            "buffer utilization, DCQCN adjusts sender rates to prevent drops."
        )
