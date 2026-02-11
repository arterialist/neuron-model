"""
FastAPI main application for the pipeline API.
"""

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.orchestrator import Orchestrator
from pipeline.api.database import Database
from pipeline.api.routes import jobs, artifacts, webhooks, uploads, job_page


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline.api")


# Initialize FastAPI app
app = FastAPI(
    title="Experiment Pipeline API",
    description="API for managing neural network experiment pipelines",
    version="0.1.0",
)

# Configuration
BASE_OUTPUT_DIR = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "./experiments"))
DB_PATH = os.environ.get("PIPELINE_DB_PATH", "pipeline.db")

# Initialize components
orchestrator = Orchestrator(BASE_OUTPUT_DIR, logger=logger)
database = Database(DB_PATH)

# Set up routes
jobs.set_orchestrator(orchestrator)
webhooks.set_database(database)

app.include_router(jobs.router)
app.include_router(artifacts.router)
app.include_router(webhooks.router)
app.include_router(uploads.router)
app.include_router(job_page.router)

# Templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/stats")
async def stats():
    """Get pipeline statistics."""
    from pipeline.api.routes.jobs import get_orchestrator

    orch = get_orchestrator()
    all_jobs = orch.list_jobs()

    status_counts = {}
    for job in all_jobs:
        status_counts[job.status] = status_counts.get(job.status, 0) + 1

    return {
        "total_jobs": len(all_jobs),
        "status_counts": status_counts,
        "output_dir": str(BASE_OUTPUT_DIR),
    }


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    orchestrator.shutdown()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
