
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "database", "warehouse.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT NOT NULL,
                created_at REAL NOT NULL,
                params_json TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                num_episodes INTEGER NOT NULL,
                final_avg_reward REAL,
                best_avg_reward REAL,
                reward_curve_json TEXT,
                model_path TEXT,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                order_id INTEGER,
                product TEXT,
                priority TEXT,
                waiting_time REAL
            )
        """)


def insert_training_run(algorithm: str, params: dict, duration_seconds: float,
                         num_episodes: int, final_avg_reward: Optional[float],
                         best_avg_reward: Optional[float], reward_curve: list,
                         model_path: Optional[str] = None, notes: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO training_runs
                (algorithm, created_at, params_json, duration_seconds, num_episodes,
                 final_avg_reward, best_avg_reward, reward_curve_json, model_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (algorithm, time.time(), json.dumps(params), duration_seconds, num_episodes,
              final_avg_reward, best_avg_reward, json.dumps(reward_curve), model_path, notes))
        return cur.lastrowid


def get_training_run(run_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_training_runs(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def latest_run_per_algorithm() -> dict:
    """Returns {algorithm: latest_run_dict} for the comparison page."""
    runs = list_training_runs(limit=200)
    latest = {}
    for r in runs:
        if r["algorithm"] not in latest:
            latest[r["algorithm"]] = r
    return latest
