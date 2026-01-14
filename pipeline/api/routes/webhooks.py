"""
Webhook management routes.
"""

import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from pipeline.api.models import WebhookCreate, WebhookInfo, ApiResponse
from pipeline.api.database import Database


router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# Database instance (will be set from main.py)
_db: Database | None = None


def set_database(db: Database) -> None:
    """Set the database instance."""
    global _db
    _db = db


def get_database() -> Database:
    """Get the database instance."""
    if _db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return _db


@router.get("", response_model=List[WebhookInfo])
async def list_webhooks():
    """List all registered webhooks."""
    # For now, return empty list as webhooks are stored in job configs
    # This endpoint could be expanded to manage global webhooks
    return []


@router.post("", response_model=WebhookInfo)
async def create_webhook(webhook: WebhookCreate):
    """Register a new webhook (not implemented - webhooks are per-job in config)."""
    raise HTTPException(
        status_code=501,
        detail="Global webhooks not implemented. Configure webhooks in job config.",
    )


@router.delete("/{webhook_id}", response_model=ApiResponse)
async def delete_webhook(webhook_id: int):
    """Delete a webhook."""
    raise HTTPException(
        status_code=501,
        detail="Global webhooks not implemented. Configure webhooks in job config.",
    )


@router.post("/test", response_model=ApiResponse)
async def test_webhook(webhook: WebhookCreate):
    """Test a webhook URL by sending a test payload."""
    import requests

    try:
        payload = {
            "event": "test",
            "message": "Webhook test from pipeline API",
            "timestamp": datetime.now().isoformat(),
        }

        response = requests.post(
            webhook.url,
            json=payload,
            headers=webhook.headers,
            timeout=10,
        )

        return ApiResponse(
            success=response.status_code < 400,
            message=f"Webhook response: {response.status_code}",
            data={"status_code": response.status_code},
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=f"Webhook test failed: {e}",
        )
