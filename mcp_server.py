"""XPU-Fabric MCP Server wrapping PromQL endpoints."""

import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="XPU-Fabric Model Context Protocol (MCP) Server")

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")


class PromQueryRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {"status": "ok", "mcp_version": "1.0.0"}


@app.post("/mcp/v1/tools/promql_query")
def execute_promql(req: PromQueryRequest):
    """
    Execute a PromQL query against the fabric's live telemetry database.
    This endpoint is exposed as an MCP 'tool' for the LLM Agent.
    """
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": req.query},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        # Simplify the response for the LLM context limits
        results = []
        if data.get("status") == "success":
            for result in data["data"]["result"]:
                metric = result.get("metric", {})
                value = result.get("value", [None, None])[1]
                
                # Format easily parseable strings: "spine2 port 4: 95.5"
                label_str = " ".join([f"{k}={v}" for k, v in metric.items() if k != "__name__"])
                results.append(f"{label_str}: {value}")
                
        return {"tool_output": "\n".join(results) if results else "No data returned."}
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Prometheus query failed: {str(e)}")


if __name__ == "__main__":
    print(f"Starting MCP Server bridging to {PROMETHEUS_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
