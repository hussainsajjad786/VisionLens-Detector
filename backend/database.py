import sqlite3
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "visionlens.db"
SESSION_DB = ContextVar("session_database", default=None)


@contextmanager
def connection():
    conn = sqlite3.connect(SESSION_DB.get() or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            patient_id TEXT,
            age INTEGER,
            eye TEXT,
            stage TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS doctor_feedback (
            screening_id INTEGER PRIMARY KEY,
            review_status TEXT NOT NULL DEFAULT 'Pending',
            doctor_name TEXT,
            notes TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (screening_id) REFERENCES screenings(id)
        )""")


def save_screening(patient_name, patient_id, age, eye, stage, confidence):
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO screenings (patient_name, patient_id, age, eye, stage, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (patient_name, patient_id, age, eye, stage, confidence),
        )
        return cursor.lastrowid


def recent_screenings(limit=10):
    with connection() as conn:
        return [dict(row) for row in conn.execute(
            """SELECT s.*, COALESCE(f.review_status, 'Pending') AS review_status,
                      COALESCE(f.doctor_name, '') AS doctor_name,
                      COALESCE(f.notes, '') AS doctor_notes,
                      f.updated_at AS reviewed_at
               FROM screenings s LEFT JOIN doctor_feedback f ON f.screening_id = s.id
               ORDER BY s.id DESC LIMIT ?""", (limit,)
        )]


def get_screening(screening_id):
    with connection() as conn:
        row = conn.execute(
            """SELECT s.*, COALESCE(f.review_status, 'Pending') AS review_status,
                      COALESCE(f.doctor_name, '') AS doctor_name,
                      COALESCE(f.notes, '') AS doctor_notes,
                      f.updated_at AS reviewed_at
               FROM screenings s LEFT JOIN doctor_feedback f ON f.screening_id = s.id
               WHERE s.id = ?""", (screening_id,),
        ).fetchone()
        return dict(row) if row else None


def patient_history(patient_id=None, patient_name=None, limit=100):
    filters, values = [], []
    if patient_id:
        filters.append("LOWER(s.patient_id) LIKE ?")
        values.append(f"%{patient_id.strip().lower()}%")
    if patient_name:
        filters.append("LOWER(s.patient_name) LIKE ?")
        values.append(f"%{patient_name.strip().lower()}%")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    values.append(limit)
    with connection() as conn:
        return [dict(row) for row in conn.execute(
            f"""SELECT s.*, COALESCE(f.review_status, 'Pending') AS review_status,
                       COALESCE(f.doctor_name, '') AS doctor_name,
                       COALESCE(f.notes, '') AS doctor_notes,
                       f.updated_at AS reviewed_at
                FROM screenings s LEFT JOIN doctor_feedback f ON f.screening_id = s.id
                {where} ORDER BY s.created_at DESC, s.id DESC LIMIT ?""", values,
        )]


def save_feedback(screening_id, review_status, doctor_name, notes):
    with connection() as conn:
        exists = conn.execute("SELECT id FROM screenings WHERE id = ?", (screening_id,)).fetchone()
        if not exists:
            return False
        conn.execute(
            """INSERT INTO doctor_feedback (screening_id, review_status, doctor_name, notes, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(screening_id) DO UPDATE SET review_status = excluded.review_status,
                   doctor_name = excluded.doctor_name, notes = excluded.notes, updated_at = CURRENT_TIMESTAMP""",
            (screening_id, review_status, doctor_name.strip(), notes.strip()),
        )
        return True


def delete_screening(screening_id):
    """Remove one screening and its linked doctor feedback, if it exists."""
    with connection() as conn:
        exists = conn.execute("SELECT id FROM screenings WHERE id = ?", (screening_id,)).fetchone()
        if not exists:
            return False
        conn.execute("DELETE FROM doctor_feedback WHERE screening_id = ?", (screening_id,))
        conn.execute("DELETE FROM screenings WHERE id = ?", (screening_id,))
        return True


def analytics_summary():
    with connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
        reviewed = conn.execute("SELECT COUNT(*) FROM doctor_feedback WHERE review_status != 'Pending'").fetchone()[0]
        stages = {row["stage"]: row["count"] for row in conn.execute(
            "SELECT stage, COUNT(*) AS count FROM screenings GROUP BY stage"
        )}
        timeline = [dict(row) for row in conn.execute(
            """SELECT substr(created_at, 1, 10) AS date, COUNT(*) AS screenings
               FROM screenings GROUP BY substr(created_at, 1, 10) ORDER BY date"""
        )]
        return {"total": total, "reviewed": reviewed, "pending": total - reviewed, "stages": stages, "timeline": timeline}
