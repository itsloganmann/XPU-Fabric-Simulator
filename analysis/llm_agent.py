"""LLM-powered network analysis with deterministic fallback."""

import json
import os

# Rule-based analysis thresholds.
HIGH_DROP_RATE = 1.0
HIGH_QUEUE_UTIL = 80.0
HIGH_TAIL_LATENCY = 20.0


def analyze_metrics(metrics_json: str, routing_mode: str = "ECMP") -> str:
    """Analyze simulation metrics using an LLM or fall back to rule-based analysis.

    Tries OpenAI first if OPENAI_API_KEY is set, otherwise uses a deterministic
    rule engine so the demo works without any API keys.

    Args:
        metrics_json: JSON string of simulation summary metrics.
        routing_mode: The routing mode used in the simulation.

    Returns:
        Human-readable analysis string.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return _llm_analysis(metrics_json, routing_mode, api_key)
    return _rule_based_analysis(metrics_json, routing_mode)


def _llm_analysis(metrics_json: str, routing_mode: str, api_key: str) -> str:
    """Send metrics to OpenAI for analysis."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Arista NetOps AI assistant. Analyze data center "
                        "network simulation metrics and explain congestion patterns, "
                        "root causes, and recommendations. Be specific about which "
                        "switches and ports are problematic. Reference the routing "
                        f"mode used ({routing_mode}) in your analysis. Keep the "
                        "response under 200 words."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze these simulation metrics:\n{metrics_json}",
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fall back to rule-based if the API call fails.
        return _rule_based_analysis(metrics_json, routing_mode) + (
            f"\n\n[Note: LLM analysis unavailable ({e}). Showing rule-based analysis.]"
        )


def _rule_based_analysis(metrics_json: str, routing_mode: str) -> str:
    """Deterministic rule-based analysis of simulation metrics."""
    data = json.loads(metrics_json)
    findings = []
    recommendations = []

    # Check overall drop rate.
    drop_rate = data.get("drop_rate_pct", 0)
    total_drops = data.get("total_packets_dropped", 0)
    total_sent = data.get("total_packets_sent", 0)

    if drop_rate > HIGH_DROP_RATE:
        findings.append(
            f"The network dropped {total_drops} of {total_sent} packets "
            f"({drop_rate}% drop rate), indicating significant congestion."
        )

    # Analyze per-switch metrics.
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
            f"Spine switches [{spine_names}] experienced buffer saturation "
            f"(peak utilization above {HIGH_QUEUE_UTIL}%)."
        )

        if routing_mode == "ECMP":
            findings.append(
                "This is consistent with ECMP hash collisions, where multiple "
                "flows hash to the same spine uplink, leaving other spines "
                "underutilized."
            )
            recommendations.append(
                "Switch to Dynamic Load Balancing (DLB) to spray packets across "
                "spines based on real-time queue depths, avoiding hash-collision "
                "hotspots."
            )

    if congested_leaves:
        leaf_names = ", ".join(s[0] for s in congested_leaves)
        findings.append(
            f"Leaf switches [{leaf_names}] show high queue utilization, "
            f"suggesting downstream bottlenecks or oversubscription."
        )

    # Analyze latency.
    latency = data.get("latency", {})
    p99 = latency.get("p99", 0)
    p50 = latency.get("p50", 0)

    if p99 > HIGH_TAIL_LATENCY:
        findings.append(
            f"Tail latency is elevated (p99: {p99}, p50: {p50}), "
            f"indicating queuing delays from congested links."
        )

    # ECN/DCQCN effectiveness.
    ecn_marks = data.get("total_ecn_marks", 0)
    if ecn_marks > 0:
        findings.append(
            f"ECN marked {ecn_marks} packets, triggering DCQCN rate limiting "
            f"on affected flows. This helped prevent additional drops but "
            f"increased flow completion time."
        )

    if not findings:
        findings.append(
            "The network performed well with no significant congestion. "
            "Queue utilization remained within acceptable limits and packet "
            "drops were minimal."
        )

    if routing_mode == "DLB" and not congested_spines:
        recommendations.append(
            "DLB is effectively distributing load across spine switches. "
            "The even queue distribution confirms dynamic spraying is working."
        )

    if not recommendations:
        recommendations.append(
            "Current configuration is performing adequately. Consider "
            "increasing buffer capacity if latency requirements tighten."
        )

    # Build the report.
    report = "## Network Analysis\n\n"
    report += "### Findings\n\n"
    for f in findings:
        report += f"- {f}\n"
    report += "\n### Recommendations\n\n"
    for r in recommendations:
        report += f"- {r}\n"

    return report
