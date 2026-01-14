"""Route for serving the job details page."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter(tags=["pages"])

templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_details_page(request: Request, job_id: str):
    """Serve the job details page."""
    return templates.TemplateResponse(
        "job_details.html", {"request": request, "job_id": job_id}
    )
