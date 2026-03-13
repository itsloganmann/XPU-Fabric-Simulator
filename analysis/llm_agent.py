"""LLM-based analysis pipeline for XPU fabric telemetry.

Parses telemetry logs, identifies bottleneck configurations automatically,
and recommends routing optimizations.
"""

import json
import os

# Bottleneck detection thresholds.
HIGH_DROP_RATE = 1.0
HIGH_QUEUE_UTIL = 80.0
HIGH_TAIL_LATENCY = 20.0


def analyze_metrics(metrics_json: str, routing_mode: str = "ECMP") -> str:
    """Analyze fabric telemetry using an LLM or fall back to rule-based analysis.

    Tries OpenAI first if OPENAI_API_KEY is set, otherwise uses a deterministic
    rule engine so the demo works without any API keys.

    Args:
        metrics_json: JSON telemetry log from the simulation.
        routing_mode: The routing mode used in the simulation.

    Returns:
        Human-readable analysis with bottleneck identification and routing recommendations.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return _llm_analysis(metrics_json, routing_mode, api_key)
    return _rule_based_analysis(metrics_json, routing_mode)


def _llm_analysis(metrics_json: str, routing_mode: str, api_key: str) -> str:
    """Send telemetry to OpenAI for bottleneck analysis."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an XPU fabric network analyst. Parse the following "
                        "telemetry logs from a CLOS fabric simulation. Identify "
                        "bottleneck configurations automatically and recommend "
                        "routing optimizations. Be specific about which switches "
                        "and ports are problematic. Reference the routing mode used "
                        f"({routing_mode}) and whether ECMP hashing is causing "
                        "collisions or whether adaptive load balancing is distributing "
                        "traffic effectively. Keep the response under 200 words."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze this telemetry log:\n{metrics_json}",
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _rule_based_analysis(metrics_json, routing_mode) + (
            f"\n\n[Note: LLM analysis unavailable ({e}). Showing rule-based analysis.]"
        )


def _rule_based_analysis(metrics_json: str, routing_mode: str) -> str:
    """Parse telemetry logs and identify bottleneck configurations."""
    data = json.loads(metrics_json)
    findings = []
    recommendations = []

    # Overall drop rate analysis.
    drop_rate = data.get("drop_rate_pct", 0)
    total_drops = data.get("total_packets_dropped", 0)
    total_sent = data.get("total_packets_sent", 0)

    if drop_rate > HIGH_DROP_RATE:
        findings.append(
            f"The fabric dropped {total_drops} of {total_sent} packets "
            f"({drop_rate}% drop rate), indicating congestion in the CLOS topology."
        )

    # Per-switch bottleneck identification.
    switches = data.get("switches", {})
    congested_spines = []
    congested_leaves = []

    for name, stats in switches.items():
        max_util = stats.get("max_queue_utilization_pct", 0)
        drops = stats.get("total_drops", 0)

        if max_util >= HIGH_QUEUE_UTIL:
            if "spine" in name:
                congested_spines.append((name, max_util, drops))
            else:
                congested_leaves.append((name, max_util, drops))

    if congested_spines:
        spine_names = ", ".join(s[0] for s in congested_spines)
        findings.append(
            f"Bottleneck identified: spine switches [{spine_names}] experienced "
            f"buffer saturation (peak utilization above {HIGH_QUEUE_UTIL}%)."
        )

        if routing_mode == "ECMP":
            findings.append(
                "ECMP hashing inefficiency detected: multiple flows hash to the "
                "same spine uplink, creating hotspots while leaving other spines "
                "underutilized. This is a known limitation of static hash-based "
                "path selection in CLOS fabrics."
            )
            recommendations.append(
                "Switch to Adaptive Load Balancing to spray packets across "
                "spines based on real-time queue depths, eliminating the "
                "hash-collision bottleneck."
            )

    if congested_leaves:
        leaf_names = ", ".join(s[0] for s in congested_leaves)
        findings.append(
            f"Bottleneck identified: leaf switches [{leaf_names}] show high queue "
            f"utilization, suggesting downstream oversubscription in the fabric."
        )

    # Latency analysis.
    latency = data.get("latency", {})
    p99 = latency.get("p99", 0)
    p50 = latency.get("p50", 0)

    if p99 > HIGH_TAIL_LATENCY:
        findings.append(
            f"Tail latency elevated (p99: {p99}, p50: {p50}), caused by "
            f"queuing delays at congested fabric links."
        )

    # ECN/DCQCN analysis.
    ecn_marks = data.get("total_ecn_marks", 0)
    if ecn_marks > 0:
        findings.append(
            f"ECN marked {ecn_marks} packets, triggering DCQCN rate limiting. "
            f"Congestion control helped prevent additional drops but increased "
            f"flow completion time."
        )

    if not findings:
        findings.append(
            "No bottlenecks detected. The CLOS fabric performed well with "
            "queue utilization within acceptable limits and minimal packet drops."
        )

    if routing_mode == "Adaptive Load Balancing" and not congested_spines:
        recommendations.append(
            "Adaptive load balancing is effectively distributing traffic across "
            "spine switches. Even queue distribution confirms congestion-free operation."
        )

    if not recommendations:
        recommendations.append(
            "Current fabric configuration is performing adequately. Consider "
            "increasing buffer capacity if latency requirements tighten."
        )

    # Build the analysis report.
    report = "## Telemetry Analysis\n\n"
    report += "### Bottleneck Identification\n\n"
    for f in findings:
        report += f"- {f}\n"
    report += "\n### Routing Optimization Recommendations\n\n"
    for r in recommendations:
        report += f"- {r}\n"

    return report
