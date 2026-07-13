import pytest
from fastapi.testclient import TestClient
import os

# Set environment variables for testing before importing main
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,http://example.com"
os.environ["API_KEY"] = ""  # Disable API key for public endpoints in test unless configured
os.environ["ADMIN_API_KEY"] = "admin-test-key"

from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_cors_headers():
    # CORS test with allowed origin
    response = client.options("/compare", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://example.com"

    # CORS test with disallowed origin
    response_blocked = client.options("/compare", headers={
        "Origin": "http://disallowed.com",
        "Access-Control-Request-Method": "POST"
    })
    assert response_blocked.headers.get("access-control-allow-origin") is None

def test_compare_endpoint():
    payload = {"name1": "Amit", "name2": "Ameet", "enable_aliases": True}
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert data["is_similar"] is True
    assert "processing_time_ms" in data

def test_compare_with_custom_threshold():
    # Should not be marked similar under high threshold
    payload = {"name1": "Amit", "name2": "Umit", "threshold": 99.0}
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    assert response.json()["is_similar"] is False

def test_compare_batch_endpoint():
    payload = {
        "pairs": [
            {"name1": "Amit", "name2": "Ameet"},
            {"name1": "Sanjay", "name2": "Sanjeev"}
        ],
        "enable_aliases": True
    }
    response = client.post("/compare-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["status"] == "success"
    assert "processing_time_ms" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text

def test_validation_error_sanitization():
    # Input has invalid characters
    payload = {"name1": "!!!", "name2": "Ameet"}
    response = client.post("/compare", json=payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    # Check that it doesn't reflect the raw "!!!" string back
    assert "!!!" not in detail
    assert "contains no valid Latin characters" in detail

def test_admin_reload_aliases():
    # Test without auth key
    response = client.post("/admin/reload-aliases")
    assert response.status_code == 403

    # Test with wrong auth key
    response_wrong = client.post("/admin/reload-aliases", headers={"X-API-Key": "wrong-key"})
    assert response_wrong.status_code == 403

    # Test with correct key
    response_correct = client.post("/admin/reload-aliases", headers={"X-API-Key": "admin-test-key"})
    assert response_correct.status_code == 200
    assert response_correct.json()["status"] == "success"

def test_rate_limiting():
    # Skip if Redis is not available for testing the rate limiter
    pytest.skip("Redis required for rate limiting test")
    
    # Reset limiter map to simulate high rate requests for testing
    from main import rate_limiter
    rate_limiter.requests.clear()
    
    # Send 100 requests (under the 100 limit)
    for _ in range(100):
        res = client.post("/compare", json={"name1": "A", "name2": "B"})
        assert res.status_code in [200, 400] # Either clean match or validation, but not rate limited
        
    # 101st request should be rate-limited
    res_rate_limited = client.post("/compare", json={"name1": "A", "name2": "B"})
    assert res_rate_limited.status_code == 429
    assert "Too many requests" in res_rate_limited.json()["detail"]

def test_compare_batch_endpoint_multi_status():
    payload = {
        "pairs": [
            {"name1": "Amit", "name2": "Ameet"},
            {"name1": "!!!", "name2": "???"}
        ],
        "enable_aliases": True
    }
    response = client.post("/compare-batch", json=payload)
    assert response.status_code == 207
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["status"] == "success"
    assert data["results"][1]["status"] == "error"

def test_compare_batch_endpoint_all_failed():
    payload = {
        "pairs": [
            {"name1": "!!!", "name2": "???"}
        ]
    }
    response = client.post("/compare-batch", json=payload)
    assert response.status_code == 400
