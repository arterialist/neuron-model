"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient

from pipeline.api.main import app
from pipeline.api.routes.jobs import set_orchestrator
from pipeline.orchestrator import Orchestrator


@pytest.fixture
def client(temp_dir):
    """Create a test client with a temporary orchestrator."""
    orchestrator = Orchestrator(temp_dir)
    set_orchestrator(orchestrator)

    with TestClient(app) as client:
        yield client, orchestrator


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        test_client, _ = client
        response = test_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestJobsEndpoint:
    """Tests for the jobs API endpoints."""

    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        test_client, _ = client
        response = test_client.get("/api/jobs")

        assert response.status_code == 200
        jobs = response.json()
        assert jobs == []

    def test_create_job(self, client, sample_config_yaml):
        """Test creating a new job."""
        test_client, _ = client
        response = test_client.post(
            "/api/jobs", json={"config_yaml": sample_config_yaml}
        )

        assert response.status_code == 200
        job = response.json()
        assert "job_id" in job
        assert job["job_name"] == "test_job"

    def test_create_job_invalid_config(self, client):
        """Test creating a job with invalid config."""
        test_client, _ = client
        response = test_client.post(
            "/api/jobs", json={"config_yaml": "invalid: yaml: content:"}
        )

        # Should return error due to missing required fields
        assert response.status_code == 400

    def test_get_job(self, client, sample_config_yaml):
        """Test getting a specific job."""
        test_client, orchestrator = client

        # Create a job via orchestrator
        from pipeline.config import load_config_from_string

        config = load_config_from_string(sample_config_yaml)
        job = orchestrator.create_job(config)

        response = test_client.get(f"/api/jobs/{job.job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job.job_id

    def test_get_nonexistent_job(self, client):
        """Test getting a nonexistent job returns 404."""
        test_client, _ = client
        response = test_client.get("/api/jobs/nonexistent")

        assert response.status_code == 404

    def test_list_jobs_with_jobs(self, client, sample_config_yaml):
        """Test listing jobs returns created jobs."""
        test_client, orchestrator = client

        # Create jobs via orchestrator
        from pipeline.config import load_config_from_string

        config = load_config_from_string(sample_config_yaml)
        job1 = orchestrator.create_job(config)
        job2 = orchestrator.create_job(config)

        response = test_client.get("/api/jobs")

        assert response.status_code == 200
        jobs = response.json()
        assert len(jobs) == 2


class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    def test_stats_empty(self, client):
        """Test stats with no jobs."""
        test_client, _ = client
        response = test_client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_jobs"] == 0

    def test_stats_with_jobs(self, client, sample_config_yaml):
        """Test stats with jobs."""
        test_client, _ = client

        # Create a job via the API
        response = test_client.post(
            "/api/jobs", json={"config_yaml": sample_config_yaml}
        )
        assert response.status_code == 200

        response = test_client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_jobs"] == 1


class TestWebUIEndpoint:
    """Tests for the web UI endpoint."""

    def test_root_returns_html(self, client):
        """Test that root returns HTML."""
        test_client, _ = client
        response = test_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Pipeline Dashboard" in response.text
