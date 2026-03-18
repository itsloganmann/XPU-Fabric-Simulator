<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/e0/Arista_Networks_logo.svg" alt="Arista Demo Quality" width="200"/>
</div>

# XPU-Fabric Simulator 🚀

A high-fidelity CLOS fabric discrete-event simulator evaluating ECMP hashing efficiency vs. Adaptive Load Balancing for RoCEv2 AI Data Center networks. Built with a stunning **Liquid Glass UI**, interactive **Three.js 3D Datacenter visualization**, a real-time **Prometheus telemetry pipeline**, and **LLM-powered PromQL** bottleneck analysis.

> "A true RoCEv2 environment utilizes flow control to pause traffic, keeping packet loss effectively at zero. This simulator models that physics engine."

### 🌟 Grafana Live Telemetry Dashboard
![Grafana Dashboard Diagram](https://raw.githubusercontent.com/grafana/grafana/main/public/img/grafana_icon.svg)
*(See `grafana/dashboards/xpu_fabric.json` for the provisioned layout. Insert local screenshot of Grafana dashboard here.)*

### 🤖 LLM-Powered PromQL Agent
![LLM Query Diagram](https://raw.githubusercontent.com/OpenAI/chatgpt-logo/main/public/images/chatgpt-logo.svg)
*(Insert GIF of Streamlit Agent detecting ECMP hash collisions via MCP server here.)*

---

## Architecture & Stack

This project strictly adheres to industry-standard networking telemetry pipelines:
- **Simulated Arista gNMI Poller** - Emits realistic switch telemetry (queue depths, link state, ECN marks).
- **Prometheus** - Time-series backbone scraping the fabric telemetry metrics.
- **Grafana** - Auto-provisioned UI with a pre-configured AI Fabric dashboard.
- **MCP Server** - A local Model Context Protocol backend enabling the LLM to query PromQL dynamically.
- **Streamlit & Three.js** - Ultra-modern presentation layer with deep blurs and Bloom emissives.
- **Docker Compose** - One-command orchestration.

## 🚀 Quickstart

No public website required. Spin up the entire infrastructure locally in seconds:

```bash
# Export your LLM token
export GEMINI_API_KEY="your_api_key"

# Stand up the cluster
docker compose up -d --build
```

### Access Points
- **Streamlit Application**: [http://localhost:8501](http://localhost:8501)
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) *(admin/admin)*
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

## Features & Verification
1. **0% Drop Baseline**: The physics engine perfectly mimics flow control. Launching in `Adaptive Load Balancing` yields zero packet loss across the cluster under standard oversubscribed loads.
2. **ECMP Hash Collisions**: Switch to `ECMP` to invoke hash collisions intentionally.
3. **Automated Analysis**: The LLM Agent queries the telemetry via the MCP Server and immediately detects the exact spine and root-cause routing failure. 

---
*Built as a high-fidelity demonstration for core data center network physics.*
