from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import shutil

router = APIRouter(prefix="/api/upload", tags=["uploads"])

UPLOAD_BASE = Path(os.environ.get("PIPELINE_OUTPUT_DIR", "./experiments")) / "uploads"
NETWORKS_DIR = UPLOAD_BASE / "networks"
CONFIGS_DIR = UPLOAD_BASE / "configs"

# Ensure directories exist
NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/network")
async def upload_network(file: UploadFile = File(...)):
    """Upload a neural network configuration file."""
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400, detail="Only .json files are allowed for networks"
        )

    file_path = NETWORKS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"success": True, "filename": file.filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")


@router.post("/config")
async def upload_config(file: UploadFile = File(...)):
    """Upload a pipeline configuration file."""
    if not (file.filename.endswith(".yaml") or file.filename.endswith(".yml")):
        raise HTTPException(
            status_code=400, detail="Only .yaml or .yml files are allowed for configs"
        )

    file_path = CONFIGS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"success": True, "filename": file.filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")
