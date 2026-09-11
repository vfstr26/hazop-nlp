"""
database.py — SQLite persistent storage for HAZOP NLP system.

Stores:
  - Sessions      : analysis runs with metadata
  - Scenarios     : all generated scenarios with current review status
  - HAZOP rows    : output rows from incident analysis
  - Audit log     : every user action (who reviewed what, when)
  - Users         : auth records (mirrors auth_config.json)

All reads/writes go through the Database class.
Schema is auto-created on first use — no migration scripts needed.

Usage:
    db = get_db()
    session_id = db.save_session(name, pid_text, username)
    db.save_scenarios(session_id, scenario_set)
    rows = db.get_scenarios(session_id, risk_level="Critical")
    db.update_scenario_status(scenario_id, "Accepted", reviewer="engineer1")
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
from loguru import logger

DB_PATH = Path(__file__).parent.parent / "data" / "hazop.db"


# ══════════════════════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    system_name     TEXT,
    pid_text        TEXT,
    created_by      TEXT,
    created_at      REAL,
    updated_at      REAL,
    status          TEXT DEFAULT 'active',
    scenario_count  INTEGER DEFAULT 0,
    stats_json      TEXT
);

CREATE TABLE IF NOT EXISTS scenarios (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    scenario_id     TEXT NOT NULL,
    headline        TEXT,
    mechanism       TEXT,
    risk_level      TEXT,
    risk_score      INTEGER,
    severity        INTEGER,
    likelihood      INTEGER,
    compound_depth  INTEGER,
    action_priority TEXT,
    status          TEXT DEFAULT 'New',
    llm_enriched    INTEGER DEFAULT 0,
    historical_ref  TEXT,
    events_json     TEXT,
    consequences_json TEXT,
    safeguard_gaps_json TEXT,
    recommendations_json TEXT,
    existing_safeguards_json TEXT,
    reviewer        TEXT,
    reviewed_at     REAL,
    reviewer_notes  TEXT,
    created_at      REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS hazop_rows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    node_id         TEXT,
    equipment       TEXT,
    chemical        TEXT,
    parameter       TEXT,
    guide_word      TEXT,
    deviation       TEXT,
    causes_json     TEXT,
    consequences_json TEXT,
    safeguards_existing_json TEXT,
    safeguards_recommended_json TEXT,
    risk_json       TEXT,
    actions_json    TEXT,
    action_priority TEXT,
    historical_ref  TEXT,
    notes           TEXT,
    created_at      REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT,
    action      TEXT,
    target_type TEXT,
    target_id   TEXT,
    detail      TEXT,
    timestamp   REAL
);

CREATE INDEX IF NOT EXISTS idx_scenarios_session  ON scenarios(session_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_risk      ON scenarios(risk_level);
CREATE INDEX IF NOT EXISTS idx_scenarios_status    ON scenarios(status);
CREATE INDEX IF NOT EXISTS idx_hazop_session       ON hazop_rows(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_user          ON audit_log(username);
"""


# ══════════════════════════════════════════════════════════════════════════════
# Database class
# ══════════════════════════════════════════════════════════════════════════════

class Database:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Sessions ──────────────────────────────────────────────────────────────

    def save_session(
        self,
        session_id:  str,
        name:        str,
        system_name: str = "",
        pid_text:    str = "",
        created_by:  str = "unknown",
    ) -> str:
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, name, system_name, pid_text,
                 created_by, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
            """, (session_id, name, system_name, pid_text,
                  created_by, now, now))
        logger.debug(f"Session saved: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self, created_by: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        with self._conn() as conn:
            if created_by:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE created_by=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (created_by, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def update_session_stats(self, session_id: str, stats: dict):
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET stats_json=?, updated_at=?, "
                "scenario_count=? WHERE session_id=?",
                (json.dumps(stats), time.time(),
                 stats.get("total", 0), session_id)
            )

    def delete_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM scenarios WHERE session_id=?",   (session_id,))
            conn.execute("DELETE FROM hazop_rows WHERE session_id=?",  (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?",    (session_id,))
        logger.info(f"Session deleted: {session_id}")

    # ── Scenarios ─────────────────────────────────────────────────────────────

    def save_scenarios(self, session_id: str, scenario_set) -> int:
        """Bulk-insert all scenarios from a ScenarioSet."""
        from src.scenario_models import ScenarioSet
        now = time.time()
        rows_inserted = 0
        with self._conn() as conn:
            # Clear existing scenarios for this session
            conn.execute("DELETE FROM scenarios WHERE session_id=?", (session_id,))
            for sc in scenario_set.scenarios:
                risk = sc.risk_level.value if hasattr(sc.risk_level, "value") else sc.risk_level
                conn.execute("""
                    INSERT INTO scenarios
                    (session_id, scenario_id, headline, mechanism,
                     risk_level, risk_score, severity, likelihood,
                     compound_depth, action_priority, status,
                     llm_enriched, historical_ref,
                     events_json, consequences_json,
                     safeguard_gaps_json, recommendations_json,
                     existing_safeguards_json, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    session_id, sc.scenario_id, sc.headline, sc.mechanism,
                    risk, sc.risk_score, sc.severity, sc.likelihood,
                    sc.compound_depth, sc.action_priority,
                    sc.status.value if hasattr(sc.status, "value") else sc.status,
                    int(sc.llm_enriched), sc.historical_precedent,
                    json.dumps([e.to_dict() if hasattr(e, "to_dict") else vars(e)
                                for e in sc.events]),
                    json.dumps(sc.consequences),
                    json.dumps(sc.safeguard_gaps),
                    json.dumps(sc.recommendations),
                    json.dumps(sc.existing_safeguards),
                    now,
                ))
                rows_inserted += 1

        self.update_session_stats(session_id, scenario_set.to_dict().get("stats", {}))
        logger.info(f"Saved {rows_inserted} scenarios for session {session_id}")
        return rows_inserted

    def get_scenarios(
        self,
        session_id:   str,
        risk_level:   Optional[str] = None,
        status:       Optional[str] = None,
        min_depth:    int = 1,
        limit:        int = 500,
        offset:       int = 0,
    ) -> list[dict]:
        query  = "SELECT * FROM scenarios WHERE session_id=? AND compound_depth>=?"
        params = [session_id, min_depth]
        if risk_level:
            query  += " AND risk_level=?"
            params.append(risk_level)
        if status:
            query  += " AND status=?"
            params.append(status)
        query += " ORDER BY risk_score DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_deserialise_scenario(dict(r)) for r in rows]

    def update_scenario_status(
        self,
        scenario_id:    str,
        status:         str,
        reviewer:       str = "",
        reviewer_notes: str = "",
    ):
        with self._conn() as conn:
            conn.execute("""
                UPDATE scenarios
                SET status=?, reviewer=?, reviewed_at=?, reviewer_notes=?
                WHERE scenario_id=?
            """, (status, reviewer, time.time(), reviewer_notes, scenario_id))
        self.log_action(reviewer, "update_scenario_status",
                        "scenario", scenario_id, status)

    def scenario_counts(self, session_id: str) -> dict:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT risk_level, COUNT(*) as cnt
                FROM scenarios WHERE session_id=?
                GROUP BY risk_level
            """, (session_id,)).fetchall()
        return {r["risk_level"]: r["cnt"] for r in rows}

    # ── HAZOP rows ─────────────────────────────────────────────────────────────

    def save_hazop_rows(self, session_id: str, rows: list[dict]) -> int:
        now = time.time()
        with self._conn() as conn:
            conn.execute("DELETE FROM hazop_rows WHERE session_id=?", (session_id,))
            for row in rows:
                risk = row.get("risk", {})
                conn.execute("""
                    INSERT INTO hazop_rows
                    (session_id, node_id, equipment, chemical,
                     parameter, guide_word, deviation,
                     causes_json, consequences_json,
                     safeguards_existing_json, safeguards_recommended_json,
                     risk_json, actions_json, action_priority,
                     historical_ref, notes, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    session_id,
                    row.get("node_id", ""),       row.get("equipment", ""),
                    row.get("chemical", ""),       row.get("parameter", ""),
                    row.get("guide_word", ""),     row.get("deviation", ""),
                    json.dumps(row.get("causes", [])),
                    json.dumps(row.get("consequences", [])),
                    json.dumps(row.get("safeguards_existing", [])),
                    json.dumps(row.get("safeguards_recommended", [])),
                    json.dumps(risk),
                    json.dumps(row.get("actions", [])),
                    row.get("action_priority", ""),
                    row.get("historical_ref", ""),
                    row.get("notes", ""),
                    now,
                ))
        logger.info(f"Saved {len(rows)} HAZOP rows for session {session_id}")
        return len(rows)

    def get_hazop_rows(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM hazop_rows WHERE session_id=? "
                "ORDER BY id", (session_id,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ["causes", "consequences", "safeguards_existing",
                          "safeguards_recommended", "actions", "risk"]:
                key = f"{field}_json" if field != "risk" else "risk_json"
                if key in d:
                    try:
                        d[field] = json.loads(d[key])
                    except Exception:
                        d[field] = []
            result.append(d)
        return result

    # ── Audit log ──────────────────────────────────────────────────────────────

    def log_action(self, username: str, action: str, target_type: str = "",
                   target_id: str = "", detail: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO audit_log
                (username, action, target_type, target_id, detail, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (username, action, target_type, target_id, detail, time.time()))

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Statistics ─────────────────────────────────────────────────────────────

    def global_stats(self) -> dict:
        with self._conn() as conn:
            sessions  = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            scenarios = conn.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0]
            reviewed  = conn.execute(
                "SELECT COUNT(*) FROM scenarios WHERE status != 'New'"
            ).fetchone()[0]
            critical  = conn.execute(
                "SELECT COUNT(*) FROM scenarios WHERE risk_level='Critical'"
            ).fetchone()[0]
        return {
            "sessions":  sessions,
            "scenarios": scenarios,
            "reviewed":  reviewed,
            "critical":  critical,
            "review_pct": round(reviewed / max(scenarios, 1) * 100, 1),
        }


def _deserialise_scenario(row: dict) -> dict:
    for field in ["events", "consequences", "safeguard_gaps",
                  "recommendations", "existing_safeguards"]:
        key = f"{field}_json"
        if key in row:
            try:
                row[field] = json.loads(row[key])
            except Exception:
                row[field] = []
    return row


# ── Singleton ──────────────────────────────────────────────────────────────────

_db_instance: Optional[Database] = None

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
