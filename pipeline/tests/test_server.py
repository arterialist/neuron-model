import pytest
from fastapi.testclient import TestClient
from pipeline.server.main import app

client = TestClient(app)

def test_list_jobs():
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_invalid_config():
    response = client.post(
        "/api/jobs",
        files={"config_file": ("config.txt", "invalid content")}
    )
    # The server might return 400 or 500 depending on exception handling
    # My implementation catches exceptions and returns 400
    assert response.status_code == 400
