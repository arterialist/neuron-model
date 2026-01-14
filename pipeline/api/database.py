"""
SQLite database for job history persistence.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Database:
    """SQLite database for storing job history and metadata."""

    def __init__(self, db_path: str = "pipeline.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                output_dir TEXT,
                current_step TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
        """)

        # Steps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                artifacts TEXT,
                metrics TEXT,
                error_message TEXT,
                start_time TEXT,
                end_time TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        # Logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        # Webhooks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS webhooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                events TEXT NOT NULL,
                headers TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def save_job(self, job_data: Dict[str, Any]) -> None:
        """Save or update a job."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO jobs 
            (id, name, config, status, output_dir, current_step, error_message, 
             created_at, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                job_data["job_id"],
                job_data.get("job_name", ""),
                json.dumps(job_data.get("config", {})),
                job_data.get("status", "pending"),
                job_data.get("output_dir", ""),
                job_data.get("current_step"),
                job_data.get("error_message"),
                job_data.get("created_at", datetime.now().isoformat()),
                job_data.get("started_at"),
                job_data.get("completed_at"),
            ),
        )

        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()

        if row:
            columns = [desc[0] for desc in cursor.description]
            job = dict(zip(columns, row))
            job["config"] = json.loads(job.get("config", "{}"))

            # Get steps
            cursor.execute(
                "SELECT * FROM steps WHERE job_id = ? ORDER BY id", (job_id,)
            )
            steps_rows = cursor.fetchall()
            step_columns = [desc[0] for desc in cursor.description]

            job["steps"] = {}
            for step_row in steps_rows:
                step = dict(zip(step_columns, step_row))
                step["artifacts"] = json.loads(step.get("artifacts") or "[]")
                step["metrics"] = json.loads(step.get("metrics") or "{}")
                job["steps"][step["step_name"]] = step

            conn.close()
            return job

        conn.close()
        return None

    def list_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List jobs with pagination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        else:
            cursor.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        jobs = []
        for row in rows:
            job = dict(zip(columns, row))
            job["config"] = json.loads(job.get("config", "{}"))
            jobs.append(job)

        conn.close()
        return jobs

    def save_step(self, job_id: str, step_name: str, step_data: Dict[str, Any]) -> None:
        """Save or update a step result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if step exists
        cursor.execute(
            "SELECT id FROM steps WHERE job_id = ? AND step_name = ?",
            (job_id, step_name),
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """
                UPDATE steps SET
                    status = ?,
                    artifacts = ?,
                    metrics = ?,
                    error_message = ?,
                    start_time = ?,
                    end_time = ?
                WHERE job_id = ? AND step_name = ?
            """,
                (
                    step_data.get("status", "pending"),
                    json.dumps(step_data.get("artifacts", [])),
                    json.dumps(step_data.get("metrics", {})),
                    step_data.get("error_message"),
                    step_data.get("start_time"),
                    step_data.get("end_time"),
                    job_id,
                    step_name,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO steps
                (job_id, step_name, status, artifacts, metrics, error_message, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    job_id,
                    step_name,
                    step_data.get("status", "pending"),
                    json.dumps(step_data.get("artifacts", [])),
                    json.dumps(step_data.get("metrics", {})),
                    step_data.get("error_message"),
                    step_data.get("start_time"),
                    step_data.get("end_time"),
                ),
            )

        conn.commit()
        conn.close()

    def add_log(self, job_id: str, level: str, message: str) -> None:
        """Add a log entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO logs (job_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        """,
            (
                job_id,
                datetime.now().isoformat(),
                level,
                message,
            ),
        )

        conn.commit()
        conn.close()

    def get_logs(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs for a job."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM logs WHERE job_id = ? ORDER BY timestamp DESC LIMIT ?",
            (job_id, limit),
        )

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        logs = [dict(zip(columns, row)) for row in rows]
        conn.close()

        return logs

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM logs WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM steps WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted
