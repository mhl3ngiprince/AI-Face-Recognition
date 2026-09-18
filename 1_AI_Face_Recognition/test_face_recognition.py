"""Tests for the face-recognition pipeline - run with:  pytest -q"""
import os
import tempfile

import cv2
import numpy as np
import pytest

import database as db
import face_engine as fe


@pytest.fixture()
def conn():
    tmp = tempfile.mkdtemp()
    return db.get_conn(os.path.join(tmp, "test.db"))


def _face_like(seed=0, w=200, h=200):
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 30, (h, w)).clip(0, 255).astype(np.uint8)
    cv2.circle(img, (w // 3, h // 3), 12, 40, -1)
    cv2.circle(img, (2 * w // 3, h // 3), 12, 40, -1)
    return img


def test_detector_runs():
    # must not raise even on a blank frame; may legitimately find 0 faces
    boxes = fe.detect_faces(np.zeros((320, 320), dtype=np.uint8))
    assert isinstance(boxes, list)


def test_quality_scores_blob():
    img = np.full((200, 200), 128, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    q = fe.face_quality(img, (50, 50, 100, 100))
    assert 0.0 <= q <= 1.0
    assert q > 0.1


def test_liveness_detects_movement():
    a = _face_like(1)
    b = a.copy()
    cv2.circle(b, (66, 66), 12, 0, -1)  # simulated blink
    score = fe.liveness_score(a, b, (0, 0, 200, 200))
    assert score is not None and score > 0


def test_liveness_flags_static_photo():
    a = _face_like(2)
    score = fe.liveness_score(a, a.copy(), (0, 0, 200, 200))
    assert score == pytest.approx(0.0)


def test_people_and_attendance_flow(conn):
    pid = db.add_person(conn, "Alice", employee_id="E1", department="Ops")
    assert pid > 0
    # add a real sample file
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "s.jpg")
    cv2.imwrite(p, _face_like(3))
    db.add_sample(conn, pid, p, 0.9)
    assert db.summary(conn)["samples"] == 1

    effect = db.log_event(conn, pid, 92.0, snapshot=None)
    assert effect["action"] == "punch_in"
    # second event same day = update, not a new punch-in
    assert db.log_event(conn, pid, 91.0)["action"] == "update"
    rows = db.attendance_report(conn, as_dict=True)
    assert len(rows) == 1 and rows[0]["name"] == "Alice"


def test_unknown_person_logged(conn):
    assert db.log_event(conn, None, 20.0)["action"] == "unknown"


def test_train_roundtrip(conn):
    pid = db.add_person(conn, "Bob")
    tmp = tempfile.mkdtemp()
    for i in range(10):
        p = os.path.join(tmp, f"{i}.jpg")
        cv2.imwrite(p, _face_like(seed=100 + i))
        db.add_sample(conn, pid, p, 0.8)
    eng = fe.FaceEngine(conn)
    n, people = _train_in_temp(eng, conn)
    assert n == 10 and people == 1


def _train_in_temp(eng, conn):
    """Train without writing the shared model file."""
    faces, labels = [], []
    for r in conn.execute("SELECT person_id,image_path FROM face_samples"):
        im = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        faces.append(cv2.resize(im, (fe.FACE_SIZE, fe.FACE_SIZE)))
        labels.append(r["person_id"])
    rec = fe._new_recognizer()
    rec.train(faces, np.array(labels))
    eng.recognizer = rec
    return len(faces), len(set(labels))


def test_tracker_debounce():
    t = fe.Tracker(confirm_frames=3, cooldown=100.0)
    assert t.seen(1, now=0.0) is False
    assert t.seen(1, now=0.1) is False
    assert t.seen(1, now=0.2) is True       # confirmed on 3rd sighting
    assert t.seen(1, now=0.3) is False      # within cooldown
    # after cooldown expires, the confirmed counter still holds, so it logs again
    assert t.seen(1, now=200.0) is True
