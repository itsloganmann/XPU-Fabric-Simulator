# XPU-Fabric Simulator: Technical Architecture

This document breaks down the technical implementation, physics engine, and infrastructure stack powering the XPU-Fabric Simulator. It is designed to demonstrate an understanding of real-world AI Datacenter networking (RoCEv2), scalable telemetry pipelines, and modern full-stack deployment.

---

## 1. Core Physics Engine (Python)

The simulation engine is a highly optimized, discrete-event loop written purely in Python (`simulator/`). It models the behavior of a CLOS network fabric down to the packet level.

### Physical Modeling
- **Entities:** Models `GPUs` (Endpoints), `Leafs` (Top of Rack switches), and `Spines` (Core switches).
- **Traffic Generation:** 
  - **All-to-All Collectives:** Synthesizes the exact communication patterns of distributed AI training (e.g. `NCCL All-Reduce`).
  - **Dynamic MTU & Flow Rates:** Packet generation scales inversely with user-selected MTU sizes (1500 vs Jumbo 9000 bytes) to accurately reflect payload impacts on queue depth processing.

### Congestion & Queueing Mechanisms
- **Priority Flow Control (PFC) & Headroom (IEEE 802.1Qbb):** The bedrock of lossless RoCEv2. Switches track queue utilization against a configurable `Headroom` watermark. If a port buffer breaches this mark, it issues a `PAUSE` upstream, trading queue drops for explicit tail-latency penalties to guarantee a 0% packet loss environment (unless explicitly disabled by the user).
- **Data Center Quantized Congestion Notification (DCQCN):** 
  - Switch queues actively monitor thresholds to perform **ECN (Explicit Congestion Notification)** marking on packet headers.
  - Endpoints receive these marks and throttle transmission rates via custom additive-increase/multiplicative-decrease (AIMD) math loops.

### Pathing Protocols 
- **ECMP (Equal-Cost Multi-Path):** Models standard hash-based routing. Demonstrates realistic hash-collision scenarios where multiple heavy AI flows hash to the same spine uplink, instantly filling buffers.
- **Adaptive Load Balancing:** Demonstrates intelligent, real-time telemetry routing where switches dynamically assign flows to the least-utilized outgoing queues.

---

## 2. Infrastructure & Telemetry Stack (Docker Compose)

The application models a true production network observability stack, packaged via `docker-compose.yml` for instant, one-click deployment.

### The Pipeline
1. **gNMI Poller (Python Daemon):**
   - A standalone container running `gnmi_poller.py`.
   - Simulates pulling raw queue depths, PFC pause intervals, and ECN marks directly from the simulated Arista switches over gNMI.
   - Hosts a local `/metrics` endpoint exposing this data via the `prometheus_client` library.
2. **Prometheus:**
   - Scrapes the gNMI Poller every 5 seconds.
   - Stores the time-series metric data (e.g., `fabric_queue_drops_total`).
3. **Grafana:**
   - Pre-provisioned via JSON dashboards (`grafana/dashboards/xpu_fabric.json`).
   - Automatically connects to Prometheus on boot.
   - Visualizes the fabric's health (Queue Deltas, Link Saturations) in a dark-mode Operation Command Center (OCC) view.

### MCP (Model Context Protocol) Server
- Runs as a standalone FastMCP HTTP service.
- Wraps complex PromQL (Prometheus Query Language) statements into standardized LLM-readable tools.
- Allows AI models to instantly query current fabric health states without needing to parse raw PromQL data manually.

---

## 3. Frontend & User Interface

The UI has been meticulously designed to mirror a true, high-stakes tactical command center.

### Streamlit Dashboard (`app.py`)
- Overhauled via deep CSS injections to achieve a "Palantir Gotham" visual language.
- Utilizes `Share Tech Mono` terminal typography, stark square framing, 1px deep wireframe grids, and high-contrast electric accents (`#00f2fe`).
- Houses complex user-toggles natively communicating with the Python Event Loop in real-time.

### Interactive 3D Fabric Render (`frontend/scene.html`)
- Built using raw `Three.js` mounted inside an HTML `iframe` communicating with the Streamlit parent via `postMessage`.
- **Post-Processing Pipeline:** Utilizes `EffectComposer` and `UnrealBloomPass` to give the CLOS topology physical light emission against a desaturated, high-contrast grid floor.
- Features a glassmorphic/tactical overlay HUD acting as an interactive node inventory, allowing users to search, hover, and highlight individual route paths through the 3D space.
