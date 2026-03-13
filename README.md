# XPU-Fabric Simulator

Scalable simulator for CLOS fabrics that evaluates ECMP hashing efficiency and adaptive load balancing under congestion-like traffic patterns. Includes an LLM-based analysis pipeline that parses telemetry logs, identifies bottleneck configurations automatically, and recommends routing optimizations.

**Live Demo**: [xpu-fabric-simulator.streamlit.app](https://xpu-fabric-simulator.streamlit.app)

## Features

- Configurable CLOS fabric topology (spine count, leaf count, XPUs per leaf, buffer depth)
- ECMP hashing efficiency evaluation with collision detection
- Adaptive load balancing using real-time queue depth signals
- ECN marking at 80% buffer utilization with DCQCN rate control
- All-to-All collective and sparse unicast traffic patterns
- Real-time Plotly charts: link utilization, queue depths, tail latency (p99)
- LLM-based telemetry analysis with bottleneck identification and routing recommendations

## Architecture

```
app.py                  Streamlit web application
simulator/
  engine.py             Priority-queue event loop
  switch.py             Leaf/Spine switch with per-port queues
  gpu.py                XPU traffic generation and flow control
  routing.py            ECMP and adaptive load balancing algorithms
  congestion.py         ECN marking and DCQCN rate adjustment
  workloads.py          All-to-All and sparse unicast generators
  metrics.py            Telemetry collection and JSON export
analysis/
  llm_agent.py          LLM analysis pipeline with rule-based fallback
tests/
  test_engine.py        Event loop tests
  test_routing.py       ECMP and adaptive LB tests
  test_congestion.py    ECN/DCQCN tests
  test_integration.py   Full simulation tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the app locally:

```bash
streamlit run app.py
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Configuration

Set `OPENAI_API_KEY` as an environment variable to enable LLM-powered telemetry analysis. Without it, the app uses a deterministic rule-based analysis engine that still identifies bottlenecks and recommends routing optimizations.

## How It Works

1. **Topology**: Builds a CLOS fabric with configurable spine/leaf switch counts and XPU endpoints.
2. **Traffic**: Generates flows between XPUs using all-to-all collective or sparse unicast patterns.
3. **Routing**: Packets traverse the fabric via ECMP (hash-based) or adaptive load balancing (shortest queue).
4. **Congestion Control**: Switches mark packets with ECN when queues exceed 80% capacity. DCQCN multiplicatively decreases sender rates in response.
5. **Telemetry**: Per-tick queue depths, link utilization, and flow latencies are collected and visualized.
6. **Analysis**: An LLM-based pipeline parses telemetry logs, identifies bottleneck configurations, and recommends routing optimizations.

## License

MIT
