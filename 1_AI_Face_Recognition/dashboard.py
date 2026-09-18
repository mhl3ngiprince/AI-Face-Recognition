"""Server-rendered security & attendance dashboard for the Face Recognition
system. No emoji (SVG icons only); every number comes from the real database.
"""
from __future__ import annotations

import datetime
import os
import sys
# make the shared dashboard kit importable when this file runs from its folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_shared"))

import dashboard_kit as kit   # noqa: E402
import database as db        # noqa: E402


def _today():
    return datetime.date.today().isoformat()


def render(conn=None) -> str:
    conn = conn or db.get_conn()
    today = _today()

    people = conn.execute(
        "SELECT p.id, p.name, p.employee_id, p.department,"
        " COUNT(s.id) samples FROM people p"
        " LEFT JOIN face_samples s ON s.person_id=p.id GROUP BY p.id"
        " ORDER BY p.name").fetchall()
    attendance = db.attendance_report(conn, today, as_dict=True)
    total_people = len(people)
    enrolled = sum(1 for p in people if p["samples"] > 0)
    present = len(attendance)
    late = sum(1 for a in attendance if "late" in (a["status"] or ""))
    absent = max(0, enrolled - present)

    # recognition events over the last 24h
    events = conn.execute(
        "SELECT e.event_at, e.confidence, e.liveness, p.name"
        " FROM recognition_events e LEFT JOIN people p ON p.id=e.person_id"
        " ORDER BY e.id DESC LIMIT 12").fetchall()

    # attendance trend: last 7 days present-count (real rows only)
    trend_dates, trend_counts = [], []
    for i in range(6, -1, -1):
        d = (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
        c = conn.execute("SELECT COUNT(*) n FROM attendance WHERE date=?",
                         (d,)).fetchone()["n"]
        trend_dates.append(d[5:])
        trend_counts.append(c)

    cards = [
        kit.card("Workforce", "person",
                 kit.stat("Enrolled people", enrolled, "", "person")
                 + kit.stat("Registered", total_people, "", "list"),
                 note="Source: people / face_samples tables"),
        kit.card("Attendance today", "calendar",
                 kit.stat("Present", present, "", "check")
                 + kit.stat("Late arrivals", late, "", "clock")
                 + kit.stat("Absent", absent, "", "person"),
                 note=f"Source: attendance table, {today}"),
        kit.card("Recognition events (latest)", "camera",
                 kit.table(["time", "person", "match", "liveness"],
                           [((e["event_at"] or "")[11:], e["name"] or "UNKNOWN",
                             f"{(100 - (e['confidence'] or 0)):.0f}%"
                             if e["confidence"] is not None else "-",
                             f"{e['liveness']:.1f}"
                             if e["liveness"] is not None else "-")
                            for e in events]),
                 span=2,
                 note="Source: recognition_events table (live camera events)"),
        kit.card("Attendance trend - last 7 days", "chart",
                 kit.bar_chart(trend_dates, trend_counts, "#22c55e"),
                 note="Source: attendance table (present per day)"),
        kit.card("Today's register", "list",
                 kit.table(["name", "id", "dept", "in", "out", "status"],
                           [(a["name"], a["employee_id"], a["department"],
                             (a["first_seen"] or "")[11:16],
                             (a["last_seen"] or "")[11:16], a["status"])
                            for a in attendance]),
                 span=2,
                 note="Source: attendance table"),
        kit.card("Enrolled identities", "person",
                 kit.table(["name", "employee id", "dept", "samples"],
                           [(p["name"], p["employee_id"], p["department"],
                             p["samples"]) for p in people]),
                 note="Source: people table"),
    ]
    summary = db.summary(conn)
    return kit.page("Face Recognition", "Security & attendance monitoring",
                    "".join(cards),
                    footer=("DB totals: " + str(summary)
                            + "  |  data source: local faces.db"))
