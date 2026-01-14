import requests
import time
import os
import tarfile
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"
CONFIG_PATH = Path(__file__).parent.parent / "example_config.yaml"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_artifacts"


def run_and_download():
    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        config_yaml = f.read()

    # 2. Submit job
    print(f"Submitting job from {CONFIG_PATH}...")
    response = requests.post(f"{BASE_URL}/jobs", json={"config_yaml": config_yaml})
    if response.status_code != 200:
        print(f"Failed to submit job: {response.text}")
        return

    job_id = response.json()["job_id"]
    print(f"Job submitted. ID: {job_id}")

    # 3. Poll for completion
    while True:
        job_response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        job_data = job_response.json()
        status = job_data["status"]
        print(f"Job Status: {status}")

        if status == "completed":
            break
        elif status == "failed":
            print(f"Job failed: {job_data.get('error_message')}")
            return

        time.sleep(5)

    # 4. Download artifacts
    print("Job completed. Downloading artifacts...")
    download_url = f"{BASE_URL}/artifacts/{job_id}/all.tar.gz"
    download_response = requests.get(download_url, stream=True)

    if download_response.status_code != 200:
        print(f"Failed to download artifacts: {download_response.text}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    job_dir = OUTPUT_DIR / job_id
    os.makedirs(job_dir, exist_ok=True)

    tar_path = job_dir / f"{job_id}_artifacts.tar.gz"

    with open(tar_path, "wb") as f:
        for chunk in download_response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Artifacts downloaded successfully to {tar_path}")

    # 5. Extract
    print(f"Extracting artifacts to {job_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=job_dir)

    print("Extraction complete.")


if __name__ == "__main__":
    run_and_download()
