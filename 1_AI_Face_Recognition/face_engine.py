"""Core face-recognition engine - real capture, quality gating, anti-spoof,
alignment, training and recognition. Shared by the CLI, the API and tests.

Nothing here is simulated: it opens real cameras/streams, reads real pixels
and produces real LBPH / Eigen / Fisher models.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np
import cascades
import database as db
from config import (CAMERA_HEIGHT, CAMERA_SOURCE, CAMERA_WIDTH, CONFIDENCE_GATE,
                    CONFIRM_FRAMES, DATASET_DIR, EVENT_COOLDOWN, FACE_SIZE,
                    LIVENESS_EPS, MIN_FACE_PX, MODEL_PATH, PROCESS_SCALE,
                    RECOGNIZER, REQUIRE_LIVENESS)

# Cascades are fetched/cached on demand (OpenCV 5 no longer bundles them).
def _detector():
    return cascades.get_detector("haarcascade_frontalface_default.xml")


def _profile_detector():
    try:
        return cascades.get_detector("haarcascade_profileface.xml")
    except Exception:
        return None


# ------------------------------------------------------------------ camera ---
def parse_source(source: str):
    """Turn the config string into whatever cv2.VideoCapture wants.

    "0" -> int index (webcam), "rtsp://..." -> URL, anything else -> filepath.
    """
    source = str(source).strip()
    if source.isdigit():
        return int(source)
    return source


def open_camera(source=None, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """Open a real capture device/stream and raise a helpful error if it fails."""
    cam = cv2.VideoCapture(parse_source(source if source is not None else CAMERA_SOURCE))
    if not cam.isOpened():
        raise RuntimeError(
            f"Could not open video source {source or CAMERA_SOURCE!r}. "
            "For a webcam set FR_CAMERA_SOURCE=0; for CCTV use an rtsp:// URL.")
    if isinstance(parse_source(source if source is not None else CAMERA_SOURCE), int):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam


# --------------------------------------------------------------- detection ---
def detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect frontal + profile faces and merge overlapping boxes."""
    boxes = []
    detectors = [(cascades.get_detector("haarcascade_frontalface_default.xml"),
                  {"scaleFactor": 1.2, "minNeighbors": 5})]
    prof = _profile_detector()
    if prof is not None:
        detectors.append((prof, {"scaleFactor": 1.2, "minNeighbors": 5}))
    for cascade, kwargs in detectors:
        if cascade.empty():
            continue
        found = cascade.detectMultiScale(gray, **kwargs)
        boxes.extend([tuple(int(v) for v in b) for b in found])
    return _merge_boxes(boxes)


def _merge_boxes(boxes, iou_thresh=0.3):
    """Non-max suppression so the same face isn't reported twice."""
    kept = []
    for b in sorted(boxes, key=lambda x: x[2] * x[3], reverse=True):
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def align_and_crop(gray, box, size=FACE_SIZE):
    """Crop, histogram-equalise and resize a face for stable recognition."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    roi = cv2.equalizeHist(roi)
    return cv2.resize(roi, (size, size))


# ----------------------------------------------------------------- quality ---
def face_quality(gray, box):
    """0..1 score combining sharpness (Laplacian var) and brightness balance."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    sharpness = min(cv2.Laplacian(roi, cv2.CV_64F).var() / 300.0, 1.0)
    mean = float(np.mean(roi))
    brightness = 1.0 - abs(mean - 128) / 128.0
    size_score = min(w / 200.0, 1.0)
    return round(0.5 * sharpness + 0.3 * brightness + 0.2 * size_score, 4)


# ------------------------------------------------------------ anti-spoofing ---
def liveness_score(prev_gray, curr_gray, box):
    """A tiny amount of real movement (blink, micro-motion) proves a live face.

    A printed photo or a still phone screen in front of the camera is
    pixel-identical frame to frame, so the mean absolute difference stays
    near zero. This is deliberately cheap and camera-agnostic.
    """
    if prev_gray is None:
        return None
    x, y, w, h = box
    a = prev_gray[y:y + h, x:x + w]
    b = curr_gray[y:y + h, x:x + w]
    if a.shape != b.shape or a.size == 0:
        return None
    return float(np.mean(cv2.absdiff(a, b)))


def is_live(score) -> bool:
    if not REQUIRE_LIVENESS:
        return True
    return score is not None and score >= LIVENESS_EPS


# --------------------------------------------------------------- recogniser ---
def _new_recognizer():
    if RECOGNIZER == "eigen":
        return cv2.face.EigenFaceRecognizer_create()
    if RECOGNIZER == "fisher":
        return cv2.face.FisherFaceRecognizer_create()
    return cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8,
                                              grid_x=8, grid_y=8)


class FaceEngine:
    """Loads the trained model + name table and recognises faces in frames."""

    def __init__(self, conn=None):
        self.conn = conn or db.get_conn()
        self.recognizer = None
        self.names: dict[int, str] = {}
        self.reload()

    def reload(self):
        self.names = {r["id"]: r["name"]
                      for r in self.conn.execute("SELECT id,name FROM people")}
        if os.path.exists(MODEL_PATH):
            rec = _new_recognizer()
            rec.read(MODEL_PATH)
            self.recognizer = rec
        else:
            self.recognizer = None
        return self

    def train(self):
        """Train on every stored sample. Returns (n_samples, n_people)."""
        rows = self.conn.execute(
            "SELECT person_id, image_path FROM face_samples").fetchall()
        faces, labels = [], []
        for r in rows:
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, (FACE_SIZE, FACE_SIZE)))
            labels.append(r["person_id"])
        if not faces:
            raise RuntimeError("No samples in the database. Run enroll first.")
        rec = _new_recognizer()
        rec.train(faces, np.array(labels))
        rec.write(MODEL_PATH)
        self.recognizer = rec
        return len(faces), len(set(labels))

    def recognise(self, gray, box):
        """Return (person_id|None, confidence_pct) for one face box."""
        if self.recognizer is None:
            return None, 0.0
        roi = align_and_crop(gray, box)
        if roi is None:
            return None, 0.0
        label, dist = self.recognizer.predict(roi)
        confidence = max(0.0, 100.0 - float(dist))
        if dist < CONFIDENCE_GATE and label in self.names:
            return int(label), confidence
        return None, confidence


# ------------------------------------------------------------------ enroll ---
@dataclass
class EnrollResult:
    name: str
    person_id: int
    saved: int = 0
    rejected_quality: int = 0
    outdir: str = ""

    def as_dict(self):
        return self.__dict__.copy()


def enroll_from_source(name, source=None, n_samples=40, min_quality=0.25,
                       employee_id=None, department=None, on_frame=None):
    """Capture *real* samples for `name` from a camera/file/stream.

    `on_frame(frame, boxes)` if given is called each frame for live preview
    (the API passes a no-op, the CLI passes an imshow). Returns EnrollResult.
    """
    conn = db.get_conn()
    pid = db.add_person(conn, name, employee_id, department)
    outdir = os.path.join(DATASET_DIR, f"p{pid}_{name}")
    os.makedirs(outdir, exist_ok=True)

    result = EnrollResult(name=name, person_id=pid, outdir=outdir)
    cam = open_camera(source)
    try:
        while result.saved < n_samples:
            ok, frame = cam.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = _maybe_scale(gray)
            scale_back = gray.shape[0] / small.shape[0]
            boxes = [(int(x * scale_back), int(y * scale_back),
                      int(w * scale_back), int(h * scale_back))
                     for (x, y, w, h) in detect_faces(small)]
            for box in boxes:
                x, y, w, h = box
                if w < MIN_FACE_PX:
                    continue
                q = face_quality(gray, box)
                if q < min_quality:
                    result.rejected_quality += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    continue
                face = align_and_crop(gray, box)
                if face is None:
                    continue
                path = os.path.join(outdir, f"{result.saved:04d}.jpg")
                cv2.imwrite(path, face)
                db.add_sample(conn, pid, path, q)
                result.saved += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if on_frame:
                on_frame(frame, boxes)
            elif _can_show():
                cv2.imshow(f"Enroll {name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cam.release()
        if not on_frame:
            cv2.destroyAllWindows()
    return result


def _maybe_scale(gray):
    if PROCESS_SCALE and PROCESS_SCALE != 1.0:
        return cv2.resize(gray, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    return gray


def _can_show():
    """True only when a GUI is available (headless servers have no display)."""
    try:
        return bool(os.environ.get("DISPLAY")) or os.name == "nt"
    except Exception:
        return False


# ------------------------------------------------------------------- event ---
@dataclass
class Tracker:
    """De-bounce layer: confirms a face over N frames and rate-limits events.

    This is what turns per-frame detections into trustworthy attendance events
    and prevents a single mistaken frame writing a false punch-in.
    """
    confirm_frames: int = CONFIRM_FRAMES
    cooldown: float = EVENT_COOLDOWN
    counts: dict = field(default_factory=dict)
    last_logged: dict = field(default_factory=dict)

    def seen(self, person_id, now):
        """Register a sighting. Returns True when attendance should be written."""
        self.counts[person_id] = self.counts.get(person_id, 0) + 1
        if self.counts[person_id] < self.confirm_frames:
            return False
        last = self.last_logged.get(person_id)
        if last is not None and now - last < self.cooldown:
            return False
        self.last_logged[person_id] = now
        return True
