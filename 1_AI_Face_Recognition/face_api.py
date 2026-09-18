"""FastAPI service for the Face Recognition system.

Exposes enrollment, training, recognition and attendance over HTTP so an
access-control gate, a kiosk or an HR dashboard can talk to it directly.

Run:  python main.py api       (or: uvicorn face_api:app --port 8001)
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import database as db
import face_engine as fe
from config import CONFIDENCE_GATE, SNAPSHOT_DIR

app = FastAPI(title="Face Recognition API", version="2.0.0")
_engine: Optional[fe.FaceEngine] = None


def engine() -> fe.FaceEngine:
    global _engine
    if _engine is None:
        _engine = fe.FaceEngine()
    return _engine


# ------------------------------------------------------------------ models ---
class PersonIn(BaseModel):
    name: str
    employee_id: Optional[str] = None
    department: Optional[str] = None


class TrainOut(BaseModel):
    samples: int
    people: int
    skipped_no_samples: list[dict]


class MatchOut(BaseModel):
    matched: bool
    name: Optional[str]
    person_id: Optional[int]
    confidence: float
    threshold: float


# ---------------------------------------------------------------- endpoints ---
@app.get("/health")
def health():
    eng = engine()
    return {"status": "ok", "model_loaded": eng.recognizer is not None,
            "people": len(eng.names), "db": db.summary(eng.conn)}


@app.post("/people")
def create_person(person: PersonIn):
    conn = db.get_conn()
    pid = db.add_person(conn, person.name, person.employee_id, person.department)
    return {"person_id": pid, "name": person.name}


@app.get("/people")
def list_people():
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT p.id, p.name, p.employee_id, p.department, COUNT(s.id) samples"
        " FROM people p LEFT JOIN face_samples s ON s.person_id=p.id"
        " GROUP BY p.id ORDER BY p.name").fetchall()
    return {"people": [dict(r) for r in rows]}


@app.post("/train")
def train():
    eng = engine()
    try:
        n, people = eng.train()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    eng.reload()
    conn = db.get_conn()
    return TrainOut(samples=n, people=people,
                    skipped_no_samples=db.unenrolled_people(conn))


@app.post("/enroll/{name}/sample")
async def upload_sample(name: str, file: UploadFile = File(...)):
    """Enroll by uploading a real photo (alternative to a live camera)."""
    conn = db.get_conn()
    pid = db.add_person(conn, name)
    data = await file.read()
    arr = cv2.imdecode(_to_np(data), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    boxes = fe.detect_faces(arr)
    if not boxes:
        raise HTTPException(status_code=422, detail="No face detected in the image.")
    box = max(boxes, key=lambda b: b[2] * b[3])
    face = fe.align_and_crop(arr, box)
    q = fe.face_quality(arr, box)
    outdir = os.path.join(fe.DATASET_DIR, f"p{pid}_{name}")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"upload_{len(os.listdir(outdir)):04d}.jpg")
    cv2.imwrite(path, face)
    db.add_sample(conn, pid, path, q)
    return {"person_id": pid, "saved": path, "quality": q}


@app.post("/recognize", response_model=MatchOut)
async def recognize(file: UploadFile = File(...)):
    """Recognise the largest face in an uploaded photo (or gate snapshot)."""
    eng = engine()
    if eng.recognizer is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")
    arr = cv2.imdecode(_to_np(await file.read()), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    boxes = fe.detect_faces(arr)
    if not boxes:
        raise HTTPException(status_code=422, detail="No face detected.")
    box = max(boxes, key=lambda b: b[2] * b[3])
    pid, conf = eng.recognise(arr, box)
    conn = db.get_conn()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    snap = os.path.join(SNAPSHOT_DIR, f"api_{file.filename or 'shot'}")
    cv2.imwrite(snap, arr)
    if pid is not None:
        db.log_event(conn, pid, conf, snap, camera="api")
    else:
        db.log_event(conn, None, conf, snap, camera="api")
    return MatchOut(matched=pid is not None,
                    name=eng.names.get(pid) if pid is not None else None,
                    person_id=pid, confidence=round(conf, 2),
                    threshold=CONFIDENCE_GATE)


@app.get("/attendance")
def attendance(date: Optional[str] = None):
    conn = db.get_conn()
    return {"date": date, "records": db.attendance_report(conn, date, as_dict=True)}


@app.post("/attendance/export")
def export(date: Optional[str] = None):
    conn = db.get_conn()
    return {"csv": db.export_csv(conn, date=date)}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Server-rendered security & attendance dashboard (SVG icons, no emoji)."""
    import dashboard as dash
    return dash.render(db.get_conn())


def _to_np(data: bytes):
    import numpy as np
    return np.frombuffer(data, np.uint8)
