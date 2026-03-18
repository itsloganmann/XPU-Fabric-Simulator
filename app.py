"""XPU-Fabric Simulator -- Streamlit web application.

Scalable simulator for CLOS fabrics that evaluates ECMP hashing efficiency
and adaptive load balancing under congestion-like traffic patterns.
"""

import random
import time

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
    page_title="XPU-Fabric Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS --

st.markdown("""
<style>
    /* Global App Background Enhancement */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(11, 15, 25) 0%, rgb(5, 7, 13) 90%);
    }

    /* Extreme Liquid Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -2px;
        animation: pulseGradient 6s ease-in-out infinite alternate;
        text-shadow: 0px 5px 20px rgba(79, 172, 254, 0.3);
    }

    @keyframes pulseGradient {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(15deg); }
    }

    .sub-header {
        color: #b0c4de;
        font-size: 1.25rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Ultimate Liquid Glass Card */
    .metric-card {
        background: rgba(16, 22, 36, 0.4);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        animation: shine 6s infinite;
    }

    @keyframes shine {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 15px 40px 0 rgba(0, 242, 254, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.2);
    }

    .metric-value {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #6a8bad;
        margin-top: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }

    /* LLM Analysis Box */
    .analysis-box {
        background: rgba(10, 15, 26, 0.6);
        backdrop-filter: blur(20px);
        border-left: 4px solid #00f2fe;
        border-radius: 16px;
        padding: 2.5rem;
        line-height: 1.8;
        font-size: 1.1rem;
        color: #e2e8f0;
        box-shadow: 0 10px 30px rgba(0, 242, 254, 0.08);
        transition: all 0.3s ease;
    }
    
    .analysis-box:hover {
        background: rgba(15, 20, 35, 0.7);
        box-shadow: 0 10px 40px rgba(0, 242, 254, 0.15);
    }

    /* Sidebar Glass */
    div[data-testid="stSidebar"] {
        background: rgba(8, 11, 18, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    
    /* Input widgets styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        border: none;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 1px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 242, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# -- Sidebar controls --

with st.sidebar:
    st.markdown("### CLOS Fabric Topology")
    num_spines = st.select_slider(
        "Spine Switches",
        options=[2, 4, 8],
        value=4,
        help="Number of spine switches in the CLOS fabric.",
    )
    num_leaves = st.select_slider(
        "Leaf Switches",
        options=[4, 8, 16],
        value=8,
        help="Number of leaf (ToR) switches.",
    )
    gpus_per_leaf = st.select_slider(
        "XPUs per Leaf",
        options=[4, 8, 16],
        value=4,
        help="Number of XPU endpoints per leaf switch.",
    )

    st.markdown("---")
    st.markdown("### QoS & Simulation Settings")
    qos_class = st.selectbox(
        "Traffic Class (QoS)",
        ["TC3 / DSCP 26 (Lossless AI)", "TC0 / Best Effort (Web)"],
        help="Simulated CoS mapping for these flows.",
    )
    routing_mode = st.selectbox(
        "Routing Protocol",
        ["Adaptive Load Balancing", "ECMP"],
        help="Adaptive Load Balancing uses real-time queue depths. ECMP uses hash-based path selection.",
    )
    workload_type = st.selectbox(
        "Traffic Pattern",
        ["All-to-All Collective", "Sparse Unicast"],
        help="All-to-All simulates distributed training collectives.",
    )
    sim_duration = st.slider(
        "Simulation Ticks",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of time steps to simulate.",
    )
    
    st.markdown("---")
    st.markdown("### Advanced RoCEv2 Tuning")
    pfc_enabled = st.toggle("Priority Flow Control (PFC)", value=True, help="IEEE 802.1Qbb: Send PAUSE frames to prevent queue drops.")
    mtu_size = st.selectbox("MTU Size (Bytes)", [1500, 4096, 9000], index=2, help="Jumbo Frames (9000) reduce packet counts and overhead.")
    buffer_capacity = st.slider(
        "Buffer Capacity (Packets)",
        min_value=32,
        max_value=512,
        value=128,
        step=32,
    )
    buffer_headroom = st.slider(
        "PFC Headroom (Packets)",
        min_value=2,
        max_value=64,
        value=16,
        step=2,
        help="Trigger PFC PAUSE when remaining buffer drops below this threshold.",
        disabled=not pfc_enabled
    )

    st.markdown("---")
    run_button = st.button("Run Simulation", type="primary", width="stretch")


# -- Header --

st.markdown('<div class="main-header">XPU-Fabric Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Evaluate ECMP hashing efficiency and adaptive load balancing across CLOS fabric topologies"
    "</div>",
    unsafe_allow_html=True,
)

# Render 3D component at the top
with st.container():
    try:
        with open("frontend/scene.html", "r", encoding="utf-8") as f:
            html_code = f.read()
            components.html(html_code, height=500, scrolling=False)
    except FileNotFoundError:
        st.warning("3D component not found (frontend/scene.html).")
        
st.markdown("---")


def build_topology(num_spines, num_leaves, gpus_per_leaf, buffer_capacity, pfc_enabled, buffer_headroom):
    """Construct the CLOS fabric topology with XPU endpoints."""
    spines = [
        Switch(switch_id=i, switch_type="spine", buffer_capacity=buffer_capacity, num_ports=num_leaves, pfc_enabled=pfc_enabled, headroom=buffer_headroom)
        for i in range(num_spines)
    ]
    leaves = [
        Switch(switch_id=i, switch_type="leaf", buffer_capacity=buffer_capacity, num_ports=num_spines + gpus_per_leaf, pfc_enabled=pfc_enabled, headroom=buffer_headroom)
        for i in range(num_leaves)
    ]
    gpus = []
    gpu_id = 0
    for leaf_id in range(num_leaves):
        for _ in range(gpus_per_leaf):
            gpus.append(GPU(gpu_id=gpu_id, leaf_id=leaf_id))
            gpu_id += 1

    gpu_to_leaf = {gpu.gpu_id: gpu.leaf_id for gpu in gpus}
    return spines, leaves, gpus, gpu_to_leaf


def run_simulation(num_spines, num_leaves, gpus_per_leaf, routing_mode,
                   workload_type, sim_duration, buffer_capacity,
                   pfc_enabled, mtu_size, buffer_headroom):
    """Execute the fabric simulation and return collected telemetry."""
    spines, leaves, gpus, gpu_to_leaf = build_topology(
        num_spines, num_leaves, gpus_per_leaf, buffer_capacity, pfc_enabled, buffer_headroom
    )
    metrics = MetricsCollector()
    engine = EventLoop()

    # Seed for reproducibility.
    random.seed(42)

    # Generate workload flows factoring in MTU packet scaling.
    base_rate = 10.0 if workload_type == "All-to-All Collective" else 5.0
    if workload_type == "All-to-All Collective":
        flows = all_to_all_workload(gpus, base_rate=base_rate, mtu=mtu_size)
    else:
        flows = web_traffic_workload(gpus, base_rate=base_rate, density=0.15, mtu=mtu_size)

    if not flows:
        return None, None, None, "No flows generated. Try a different configuration."

    all_switches = spines + leaves
    use_adaptive_lb = routing_mode == "Adaptive Load Balancing"

    # Run the simulation tick by tick.
    for tick in range(sim_duration):
        packets_this_tick = []

        # Each flow generates packets based on its current rate.
        for flow in flows:
            num_packets = int(flow.rate)
            if random.random() < (flow.rate - num_packets):
                num_packets += 1

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

        # Route each packet through the CLOS fabric.
        for flow, pkt in packets_this_tick:
            src_leaf = leaves[pkt.src_leaf]
            dst_leaf = leaves[pkt.dst_leaf]

            # Intra-leaf traffic skips the spine layer.
            if pkt.src_leaf == pkt.dst_leaf:
                latency = 1.0 + src_leaf.queue_depth(0) * 0.1
                flow.record_latency(latency)
                metrics.record_latency(flow.flow_id, latency)
                flow.packets_received += 1
                continue

            # Select a spine uplink.
            if use_adaptive_lb:
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
            status_up = src_leaf.enqueue(uplink_port, pkt)
            if status_up == "DROP":
                metrics.total_packets_dropped += 1
                continue
            elif status_up == "PAUSE":
                # Simulated PFC pause logic, packet is held at the sender logic temporarily
                flow.record_latency(2.0)

            # Check ECN at source leaf.
            if check_ecn(src_leaf.queue_depth(uplink_port), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the spine's ingress port.
            spine_port_in = pkt.src_leaf
            status_spin = spine.enqueue(spine_port_in, pkt)
            if status_spin == "DROP":
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                continue
            elif status_spin == "PAUSE":
                flow.record_latency(2.0)

            # Check ECN at spine ingress.
            if check_ecn(spine.queue_depth(spine_port_in), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the spine's egress port toward the destination leaf.
            spine_port_out = pkt.dst_leaf
            status_spout = spine.enqueue(spine_port_out, pkt)
            if status_spout == "DROP":
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                spine.dequeue(spine_port_in)
                continue
            elif status_spout == "PAUSE":
                flow.record_latency(2.0)

            # Check ECN at spine egress.
            if check_ecn(spine.queue_depth(spine_port_out), buffer_capacity):
                pkt.ecn_marked = True
                metrics.total_ecn_marks += 1

            # Enqueue on the destination leaf's downlink.
            downlink_port = spine_idx
            status_down = dst_leaf.enqueue(downlink_port, pkt)
            if status_down == "DROP":
                metrics.total_packets_dropped += 1
                src_leaf.dequeue(uplink_port)
                spine.dequeue(spine_port_in)
                spine.dequeue(spine_port_out)
                continue
            elif status_down == "PAUSE":
                flow.record_latency(2.0)

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

        # Record per-tick telemetry.
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

    # Summarize PFC Metrics explicitly
    total_pfc_pauses = sum(sw.total_pauses for sw in all_switches)

    summary_json = metrics.summary_json(all_switches)
    import json
    data = json.loads(summary_json)
    data["total_pfc_pauses"] = total_pfc_pauses
    
    return metrics, all_switches, json.dumps(data), None


def create_charts(metrics, all_switches):
    """Build Plotly charts from simulation telemetry."""
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

    # Link utilization over time (aggregate by switch).
    link_df = df[df["type"] == "link"].copy()
    if not link_df.empty:
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

    # Queue depths over time (top switches by peak depth).
    queue_df = df[df["type"] == "queue"].copy()
    if not queue_df.empty:
        agg_q = queue_df.groupby(["tick", "switch_name"])["queue_depth"].max().reset_index()
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

    # Tail latency (p99) over time.
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
    with st.spinner("Running high-fidelity simulation..."):
        start_time = time.time()
        metrics, switches, summary_json, error = run_simulation(
            num_spines=num_spines,
            num_leaves=num_leaves,
            gpus_per_leaf=gpus_per_leaf,
            routing_mode=routing_mode,
            workload_type=workload_type,
            sim_duration=sim_duration,
            buffer_capacity=buffer_capacity,
            pfc_enabled=pfc_enabled,
            mtu_size=mtu_size,
            buffer_headroom=buffer_headroom
        )
        elapsed = time.time() - start_time

    if error:
        st.error(error)
    else:
        # Summary telemetry cards.
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
                f'<div class="metric-label">Queue Drops</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{summary.get("total_pfc_pauses", 0):,}</div>'
                f'<div class="metric-label">PFC Pauses</div></div>',
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

        # LLM analysis panel.
        st.markdown("### Telemetry Analysis")
        with st.spinner("Analyzing telemetry..."):
            analysis = analyze_metrics(summary_json, routing_mode)
        st.markdown(
            f'<div class="analysis-box">{analysis}</div>',
            unsafe_allow_html=True,
        )

        # Expandable raw telemetry.
        with st.expander("Raw Telemetry Log (JSON)"):
            st.code(summary_json, language="json")

else:
    # Default landing state.
    st.info("Configure the CLOS fabric topology in the sidebar and click **Run Simulation** to begin.\n\n*Demo Narrative: Leave default as DLB to view a perfectly simulated lossless fabric baseline with 0% drops. Then, toggle to ECMP routing to inject hash-collision bottlenecks and observe AI inference recommendations.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "#### CLOS Fabric\n"
            "Simulate scalable CLOS topologies with configurable "
            "spine and leaf switch counts, buffer depths, and XPU endpoints."
        )
    with col2:
        st.markdown(
            "#### ECMP vs Adaptive LB\n"
            "Evaluate ECMP hashing efficiency against adaptive load balancing "
            "that distributes traffic based on real-time congestion signals."
        )
    with col3:
        st.markdown(
            "#### ECN/DCQCN\n"
            "Observe congestion-like traffic patterns in action: ECN marks packets at 80% "
            "buffer utilization, DCQCN adjusts sender rates to prevent drops."
        )
