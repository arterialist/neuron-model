"""
Database module for pipeline persistence.
"""

import os
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Default DB path: project_root/pipeline_db/pipeline.db (works for local runs)
# Override with PIPELINE_DB_PATH or PIPELINE_DB_URL
_default_db_dir = Path(__file__).resolve().parent.parent / "pipeline_db"
_default_db_path = str(_default_db_dir / "pipeline.db")

# Get DB path/URL from env
# Default to SQLite if not provided, but prefer PIPELINE_DB_URL for full connection string
DATABASE_URL = os.getenv("PIPELINE_DB_URL")
if not DATABASE_URL:
    DB_PATH = os.getenv("PIPELINE_DB_PATH", _default_db_path)
    DATABASE_URL = f"sqlite:///{DB_PATH}"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class JobModel(Base):
    """SQLAlchemy model for a pipeline job."""

    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)
    job_name = Column(String, index=True)
    status = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Store paths and configs as strings/JSON
    output_dir = Column(String)
    config_json = Column(Text)
    state_json = Column(Text)  # Full serialized job state


def init_db():
    """Initialize database tables."""
    # Ensure directory exists if using SQLite
    if DATABASE_URL.startswith("sqlite:///"):
        # Extract path from URL
        db_path = DATABASE_URL.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    Base.metadata.create_all(bind=engine)
