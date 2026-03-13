# Data Center Network Simulator

Discrete-event simulation of Leaf-Spine data center fabrics with ECMP, Dynamic Load Balancing (DLB), and ECN/DCQCN congestion control. Built with Streamlit for interactive visualization.

**Live Demo**: [discrete-event-simulator.streamlit.app](https://discrete-event-simulator.streamlit.app)

## Features

- Configurable Leaf-Spine topology (spine count, leaf count, GPUs per leaf, buffer depth)
- ECMP hash-based routing vs. Dynamic Load Balancing (DLB) comparison
- ECN marking at 80% buffer utilization with DCQCN rate control
- All-to-All (AI training) and Web Traffic workload patterns
- Real-time Plotly charts: link utilization, queue depths, tail latency (p99)
- AI-powered network analysis (OpenAI integration with rule-based fallback)

## Architecture

```
app.py                  Streamlit web application
simulator/
  engine.py             Priority-queue event loop
  switch.py             Leaf/Spine switch with per-port queues
  gpu.py                GPU traffic generation and flow control
  routing.py            ECMP and DLB routing algorithms
  congestion.py         ECN marking and DCQCN rate adjustment
  workloads.py          All-to-All and Web Traffic generators
  metrics.py            Metrics collection and JSON export
analysis/
  llm_agent.py          LLM analysis with deterministic fallback
tests/
  test_engine.py        Event loop tests
  test_routing.py       ECMP and DLB tests
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

Set `OPENAI_API_KEY` as an environment variable to enable LLM-powered analysis. Without it, the app uses a deterministic rule-based analysis engine.

## How It Works

1. **Topology**: Builds a Leaf-Spine fabric with configurable switch counts and GPU endpoints.
2. **Traffic**: Generates flows between GPUs based on the selected workload pattern.
3. **Routing**: Each packet is routed through a spine switch selected by ECMP (hash) or DLB (shortest queue).
4. **Congestion Control**: Switches mark packets with ECN when queues exceed 80% capacity. Receivers trigger DCQCN, which multiplicatively decreases the sender's rate.
5. **Metrics**: Per-tick queue depths, link utilization, and flow latencies are recorded and visualized.
6. **Analysis**: Simulation results are analyzed by an LLM or rule engine to surface congestion root causes and recommendations.

## License

MIT
