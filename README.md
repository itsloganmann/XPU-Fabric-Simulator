# XPU-Fabric Simulator

An industry-standard discrete-event simulator for evaluating ECMP hashing efficiency versus Adaptive Load Balancing in RoCEv2 AI Data Center networks.

This project accurately models a lossless physics engine where flow control tightens the ECN/DCQCN feedback loop, keeping packet loss effectively at zero under appropriate loads. It features a complete telemetry pipeline, live dashboarding, and an LLM-powered telemetry agent.

## Architecture

The simulation is encapsulated in a Docker Compose stack utilizing production-grade monitoring tools:
*   **gNMI Poller Simulator:** Emits realistic CLOS fabric telemetry (queue depths, link state, ECN marks).
*   **Prometheus:** Time-series database scraping fabric metrics.
*   **Grafana:** Auto-provisioned dashboard for live telemetry visualization.
*   **MCP Server (Model Context Protocol):** Bridging the frontend LLM agent to PromQL for live data querying.
*   **Streamlit & Three.js:** A scalable, modern frontend featuring interactive 3D datacenter rendering and dynamic metrics.

## Quickstart

Spin up the entire telemetry stack and simulation engine locally:

```bash
# Export API key for the telemetry analysis agent (Gemini or OpenAI)
export GEMINI_API_KEY="your_api_key_here"

# Launch the stack
docker compose up -d --build
```

### Endpoints
*   **Simulator UI:** [http://localhost:8501](http://localhost:8501)
*   **Grafana Telemetry:** [http://localhost:3000](http://localhost:3000) *(User: admin / Pass: admin)*
*   **Prometheus:** [http://localhost:9090](http://localhost:9090)

## Demo Scenarios

1.  **Lossless Baseline (DLB):**
    The engine initializes with **Adaptive Load Balancing**. Under this default state, the physics engine perfectly mimics RoCEv2 flow control. Queue utilization scales, but packet drops remain at 0% across the fabric.
2.  **ECMP Hash Collisions:**
    Toggle the routing mode in the UI to **ECMP**. This simulates hash collisions across the spine tier. The LLM Agent, querying metrics via the MCP server, will automatically detect and report the resulting queue saturation and latency spikes.

---
*Developed for demonstrations of core data center networking physics and telemetry architecture.*
