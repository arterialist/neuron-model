import os
import sys
import glob
import json
import logging
import subprocess
import shutil
import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

# Setup app
app = FastAPI(title="SNN Pipeline Orchestrator")

# Enable CORS
# Restrict origins for production security
# For development, use http://localhost:8000 or your frontend's actual origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Replace with your frontend's actual origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EXPERIMENTS_DIR = "experiments"
CONFIGS_DIR = "configs"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

# Global state for running jobs: job_id -> Popen object
running_jobs: Dict[str, subprocess.Popen] = {}

class JobInfo(BaseModel):
    job_id: str
    status: str # "running", "completed", "failed", "unknown"
    start_time: str
    config_name: Optional[str]
    pid: Optional[int]

class JobDetail(JobInfo):
    log_tail: List[str]
    artifacts: List[str]

@app.get("/api/jobs", response_model=List[JobInfo])
def list_jobs():
    """List all jobs found in experiments directory."""
    jobs = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return []

    for entry in os.scandir(EXPERIMENTS_DIR):
        if entry.is_dir():
            job_id = entry.name
            status = "unknown"

            # Check if running in our memory
            if job_id in running_jobs:
                proc = running_jobs[job_id]
                if proc.poll() is None:
                    status = "running"
                else:
                    # Clean up finished job from memory
                    del running_jobs[job_id]
                    status = "completed" if proc.returncode == 0 else "failed"
            else:
                # Infer status from artifacts/logs if not in memory
                # This is a heuristic.
                log_path = os.path.join(entry.path, "pipeline.log")
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            # Read log content robustly
                            lines = f.readlines()
                            if not lines:
                                status = "unknown"
                            elif any("Pipeline finished successfully" in line for line in lines[-20:]):
                                status = "completed"
                            elif any("Pipeline failed" in line for line in lines[-20:]):
                                status = "failed"
                            else:
                                status = "stopped" # or interrupted
                    except:
                        status = "unknown"

            start_parts = job_id.split("_")
            start_time = "unknown"
            if len(start_parts) >= 3:
                start_time = start_parts[-2] + "_" + start_parts[-1]

            jobs.append(JobInfo(
                job_id=job_id,
                status=status,
                start_time=start_time,
                config_name="config.yaml", # we assume standard name
                pid=running_jobs.get(job_id).pid if job_id in running_jobs else None
            ))

    # Sort by name (timestamp) descending
    jobs.sort(key=lambda x: x.job_id, reverse=True)
    return jobs

@app.post("/api/jobs")
async def create_job(config_file: UploadFile = File(...)):
    """Start a new job with an uploaded config file."""
    # 1. Validate YAML
    try:
        content = await config_file.read()
        cfg = yaml.safe_load(content)
        job_name = cfg.get("name", "experiment")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # 2. Save temporary config
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_config_path = os.path.join(CONFIGS_DIR, f"{job_name}_{timestamp}.yaml")
    with open(temp_config_path, 'wb') as f:
        f.write(content)

    # 3. Create Job Directory Explicitly
    job_id = f"{job_name}_{timestamp}"
    job_dir = os.path.join(EXPERIMENTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 4. Run pipeline with explicit output dir
    cmd = [
        sys.executable,
        "pipeline/runner.py",
        "--config", temp_config_path,
        "--job-dir", job_dir
    ]

    try:
        # Redirect stdout and stderr to log files for debugging
        log_out_path = os.path.join(job_dir, "runner.out")
        log_err_path = os.path.join(job_dir, "runner.err")
        log_out_file = open(log_out_path, "w")
        log_err_file = open(log_err_path, "w")
        process = subprocess.Popen(
            cmd,
            stdout=log_out_file,
            stderr=log_err_file
        )
        running_jobs[job_id] = process
        # Note: File handles will be closed when the process completes
        # For production, consider tracking file handles and closing them explicitly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start runner: {e}")

    return {"job_id": job_id, "status": "started", "path": job_dir}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_path = os.path.join(EXPERIMENTS_DIR, job_id)
    if not os.path.exists(job_path):
        raise HTTPException(404, "Job not found")

    # Read log tail
    log_path = os.path.join(job_path, "pipeline.log")
    log_tail = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                # Efficient tail
                f.seek(0, 2)
                fsize = f.tell()
                # Read last 20KB
                f.seek(max(fsize - 20000, 0), 0)
                log_tail = f.readlines()
        except:
            log_tail = ["Error reading log"]

    # List artifacts
    artifacts = []
    for root, dirs, files in os.walk(job_path):
        for f in files:
            rel_path = os.path.relpath(os.path.join(root, f), job_path)
            if rel_path != "pipeline.log":
                artifacts.append(rel_path)

    # Determine status
    status = "unknown"
    if job_id in running_jobs:
        if running_jobs[job_id].poll() is None:
            status = "running"
        else:
            status = "completed" if running_jobs[job_id].returncode == 0 else "failed"
    else:
        # Check log for completion
        if any("Pipeline finished successfully" in l for l in log_tail[-20:]):
            status = "completed"
        elif any("Pipeline failed" in l for l in log_tail[-20:]):
            status = "failed"
        else:
            status = "stopped"

    start_parts = job_id.split("_")
    start_time = "unknown"
    if len(start_parts) >= 3:
        start_time = start_parts[-2] + "_" + start_parts[-1]

    return JobDetail(
        job_id=job_id,
        status=status,
        start_time=start_time,
        config_name="config.yaml",
        pid=running_jobs[job_id].pid if job_id in running_jobs else None,
        log_tail=log_tail,
        artifacts=artifacts
    )

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id in running_jobs:
        proc = running_jobs[job_id]
        if proc.poll() is None:
            proc.terminate() # Kill process
            try:
                proc.wait(timeout=5)
            except:
                proc.kill()
        del running_jobs[job_id]

    job_path = os.path.join(EXPERIMENTS_DIR, job_id)
    if os.path.exists(job_path):
        shutil.rmtree(job_path)

    return {"status": "deleted"}

@app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_artifact(job_id: str, artifact_path: str):
    from fastapi.responses import FileResponse
    path = os.path.join(EXPERIMENTS_DIR, job_id, artifact_path)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)

# Serve static frontend
app.mount("/", StaticFiles(directory="pipeline/server/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
