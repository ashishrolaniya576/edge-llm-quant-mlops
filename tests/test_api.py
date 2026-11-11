"""
Unit and Integration Tests for FastAPI Edge Serving Endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serve import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing a test client with application lifespan management."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_success(client: TestClient) -> None:
    """Verify GET /health returns 200 OK and expected monitoring metrics."""
    response = client.get("/health")
    assert (
        response.status_code == 200
    ), f"Health check failed with {response.status_code}: {response.text}"

    data = response.json()
    assert "status" in data
    assert data["status"] in ("healthy", "degraded", "initializing")
    assert "model_loaded" in data
    assert "model_format" in data
    assert "memory_rss_mb" in data
    assert "memory_percent" in data
    assert "uptime_seconds" in data
    assert "threads" in data

    assert isinstance(data["memory_rss_mb"], (int, float))
    assert data["memory_rss_mb"] > 0
    assert data["threads"] >= 1


def test_analyze_telemetry_valid_oom_trace(client: TestClient) -> None:
    """Verify POST /v1/telemetry/analyze accurately processes an OOM log trace."""
    payload = {
        "log_trace": (
            "kernel: [104234.12] Out of memory: Killed process 38412 (python3) "
            "total-vm:67108864kB, anon-rss:62914560kB, file-rss:4096kB\n"
            "kernel: [104234.13] memory: usage 67108864kB, limit 67108864kB, failcnt 120"
        ),
        "system_context": "Production Kubernetes worker node edge-01",
        "temperature": 0.1,
        "max_tokens": 128,
    }

    response = client.post("/v1/telemetry/analyze", json=payload)
    assert response.status_code == 200, f"Inference request failed: {response.text}"

    data = response.json()
    assert "root_cause" in data and len(data["root_cause"]) > 5
    assert "severity" in data and data["severity"] in ("CRITICAL", "HIGH", "MEDIUM")
    assert "remediation" in data and len(data["remediation"]) > 5
    assert "metrics" in data

    metrics = data["metrics"]
    assert "ttft_ms" in metrics
    assert "total_latency_ms" in metrics
    assert "tokens_per_second" in metrics
    assert "prompt_tokens" in metrics
    assert "completion_tokens" in metrics
    assert metrics["total_latency_ms"] > 0


def test_analyze_telemetry_valid_pcie_trace(client: TestClient) -> None:
    """Verify POST /v1/telemetry/analyze processes PCIe AER failure logs."""
    payload = {
        "log_trace": (
            "kernel: [3021.11] pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: 0000:01:00.0\n"
            "kernel: [3021.12] 0000:01:00.0 PCIe Bus Error: severity=Fatal, device has fallen off the bus."
        ),
        "temperature": 0.0,
        "max_tokens": 64,
    }

    response = client.post("/v1/telemetry/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["severity"] == "CRITICAL"
    assert "pcie" in data["root_cause"].lower() or "bus" in data["root_cause"].lower()


def test_analyze_telemetry_missing_log_trace(client: TestClient) -> None:
    """Verify POST /v1/telemetry/analyze rejects requests lacking required log_trace field."""
    response = client.post("/v1/telemetry/analyze", json={})
    assert response.status_code == 422, "Expected 422 Unprocessable Entity for missing payload."


def test_analyze_telemetry_too_short_log_trace(client: TestClient) -> None:
    """Verify POST /v1/telemetry/analyze rejects traces under min_length validation threshold."""
    response = client.post("/v1/telemetry/analyze", json={"log_trace": "error"})
    assert response.status_code == 422, "Expected 422 for trace under min_length=10."


def test_analyze_telemetry_invalid_temperature_bounds(client: TestClient) -> None:
    """Verify temperature values outside [0.0, 2.0] are rejected."""
    payload = {
        "log_trace": "kernel: [100.0] Out of memory: Killed process 1024 (app)",
        "temperature": 3.5,
    }
    response = client.post("/v1/telemetry/analyze", json=payload)
    assert response.status_code == 422
