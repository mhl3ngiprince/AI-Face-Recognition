"""SQLite persistence: people, face samples, recognition events, attendance.

Also owns the *attendance policy*: turning a stream of raw recognition
events into one reliable punch-in / punch-out per person per day, with
late-arrival and early-departure flags.
"""
import csv
import datetime
import os
import sqlite3

from config import DB_PATH, EXPORT_DIR, LATE_AFTER, SHIFT_END

SCHEMA = """
CREATE TABLE IF NOT EXISTS people(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    employee_id TEXT,
    department TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS face_samples(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    quality REAL,
    captured_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS recognition_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    confidence REAL,
    liveness REAL,
    snapshot_path TEXT,
    camera TEXT,
    event_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES people(id),
    date TEXT NOT NULL,
    first_seen TEXT,
    last_seen TEXT,
    status TEXT DEFAULT 'present',
    UNIQUE(person_id, date)
);
CREATE INDEX IF NOT EXISTS idx_events_at ON recognition_events(event_at);
CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(date);
"""


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # safe concurrent reader/writer
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn
# Columns added after the first release. `CREATE TABLE IF NOT EXISTS` will not
# alter an existing faces.db, so an older database would be missing these.
_MIGRATIONS = {
    "people": {"employee_id": "TEXT", "department": "TEXT"},
    "face_samples": {"quality": "REAL"},
    "recognition_events": {"liveness": "REAL", "camera": "TEXT"},
    "attendance": {"status": "TEXT DEFAULT 'present'"},
}


def _migrate(conn):
    for table, cols in _MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue
        for name, decl in cols.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def add_person(conn, name, employee_id=None, department=None):
    conn.execute(
        "INSERT INTO people(name, employee_id, department) VALUES(?,?,?)"
        " ON CONFLICT(name) DO UPDATE SET"
        " employee_id=COALESCE(excluded.employee_id, people.employee_id),"
        " department =COALESCE(excluded.department,  people.department)",
        (name, employee_id, department))
    conn.commit()
    return conn.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()["id"]


def add_sample(conn, person_id, path, quality=None):
    conn.execute("INSERT INTO face_samples(person_id,image_path,quality) VALUES(?,?,?)",
                 (person_id, path, quality))
    conn.commit()


def has_sample(conn, person_id, path):
    row = conn.execute("SELECT 1 FROM face_samples WHERE person_id=? AND image_path=?",
                       (person_id, path)).fetchone()
    return row is not None


def unenrolled_people(conn):
    """People with zero samples - these are ignored by training."""
    rows = conn.execute(
        "SELECT p.id, p.name FROM people p"
        " LEFT JOIN face_samples s ON s.person_id = p.id"
        " GROUP BY p.id HAVING COUNT(s.id) = 0").fetchall()
    return [dict(r) for r in rows]


def log_event(conn, person_id, confidence, snapshot=None, liveness=None, camera=None):
    """Record a raw recognition event and fold it into today's attendance row.

    Returns a dict describing the effect so callers can show a kiosk message:
        {"action": "punch_in"|"update"|"unknown", "status": "present"|"late"}
    """
    conn.execute(
        "INSERT INTO recognition_events"
        "(person_id,confidence,liveness,snapshot_path,camera)"
        " VALUES(?,?,?,?,?)", (person_id, confidence, liveness, snapshot, camera))
    if person_id is None:
        conn.commit()
        return {"action": "unknown"}

    today = datetime.date.today().isoformat()
    now = _now()
    existing = conn.execute(
        "SELECT id, first_seen FROM attendance WHERE person_id=? AND date=?",
        (person_id, today)).fetchone()
    if existing is None:
        status = "late" if now[11:16] > LATE_AFTER else "present"
        conn.execute(
            "INSERT INTO attendance(person_id,date,first_seen,last_seen,status)"
            " VALUES(?,?,?,?,?)", (person_id, today, now, now, status))
        conn.commit()
        return {"action": "punch_in", "status": status}
    conn.execute("UPDATE attendance SET last_seen=? WHERE id=?", (now, existing["id"]))
    conn.commit()
    return {"action": "update"}


def mark_departures(conn, date=None):
    """Flag anyone whose last_seen is before SHIFT_END as an early departure."""
    date = date or datetime.date.today().isoformat()
    for r in conn.execute("SELECT id, last_seen, status FROM attendance WHERE date=?",
                          (date,)).fetchall():
        if (r["last_seen"] and r["last_seen"][11:16] < SHIFT_END
                and "early" not in (r["status"] or "")):
            conn.execute("UPDATE attendance SET status=? WHERE id=?",
                         (f"{r['status'] or 'present'}+early", r["id"]))
    conn.commit()


def attendance_report(conn, date=None, as_dict=False):
    date = date or datetime.date.today().isoformat()
    rows = conn.execute(
        "SELECT p.name, p.employee_id, p.department, a.date, a.first_seen,"
        " a.last_seen, a.status FROM attendance a"
        " JOIN people p ON p.id=a.person_id WHERE a.date=?"
        " ORDER BY a.first_seen", (date,)).fetchall()
    data = [dict(r) for r in rows]
    return data if as_dict else _print_rows(data)


def _print_rows(data):
    if not data:
        print("(no attendance records for the requested day)")
        return []
    print(f"{'name':<20}{'emp_id':<12}{'dept':<14}{'in':<10}{'out':<10}status")
    for r in data:
        print(f"{r['name']:<20}{(r['employee_id'] or '-'):<12}"
              f"{(r['department'] or '-'):<14}{r['first_seen'][11:]:<10}"
              f"{(r['last_seen'] or '')[11:]:<10}{r['status']}")
    return data


def export_csv(conn, out_dir=EXPORT_DIR, date=None):
    """Write a day's attendance to a CSV that payroll/HR systems can import."""
    os.makedirs(out_dir, exist_ok=True)
    date = date or datetime.date.today().isoformat()
    data = attendance_report(conn, date, as_dict=True)
    path = os.path.join(out_dir, f"attendance_{date}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "employee_id", "department",
                                           "date", "first_seen", "last_seen", "status"])
        w.writeheader()
        w.writerows(data)
    print(f"Wrote {len(data)} rows -> {path}")
    return path


def summary(conn):
    c = conn.execute("SELECT COUNT(*) n FROM people").fetchone()["n"]
    s = conn.execute("SELECT COUNT(*) n FROM face_samples").fetchone()["n"]
    e = conn.execute("SELECT COUNT(*) n FROM recognition_events").fetchone()["n"]
    return {"people": c, "samples": s, "events": e}


# backward-compatible alias used by older scripts
report = attendance_report
