"""AI Face Recognition - production security & attendance system.

A real video pipeline with no simulation:
    webcam / video file / RTSP stream
        -> face detection (frontal + profile, merged)
        -> quality gate -> liveness (anti-photo)
        -> LBPH / Eigen / Fisher recognition
        -> debounced attendance in SQLite + snapshot evidence + CSV export

Commands
--------
    python main.py enroll  "Alice" [--samples 40] [--source 0]
    python main.py enroll-batch                                        # bulk, interactive
    python main.py train
    python main.py run     [--source 0] [--no-display] [--seconds 0]
    python main.py report  [--date YYYY-MM-DD] [--csv]
    python main.py people
    python main.py api                                                 # start REST service
    python main.py selftest                                            # offline pipeline check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

import database as db
import face_engine as fe
from config import (CAMERA_SOURCE, CONFIDENCE_GATE, LATE_AFTER, SHIFT_END,
                    SHIFT_START, SNAPSHOT_DIR)


# --------------------------------------------------------------- utilities ---
def _print_banner(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def _bounds(box, frame_shape):
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    return x0, y0, x1, y1


# ---------------------------------------------------------------- enroll -----
def cmd_enroll(args):
    result = fe.enroll_from_source(
        args.name,
        source=args.source,
        n_samples=args.samples,
        employee_id=args.employee_id,
        department=args.department)
    _print_banner(f"Enrollment complete: {result.name} (#{result.person_id})")
    print(f"  samples saved     : {result.saved}")
    print(f"  rejected (blurry) : {result.rejected_quality}")
    print(f"  dataset folder    : {result.outdir}")
    if result.saved:
        print("\nNext: python main.py train")


def cmd_enroll_batch(args):
    """Enroll several people in one camera session - real HR onboarding."""
    conn = db.get_conn()
    print("Bulk enrollment. Leave the name blank to finish.")
    while True:
        name = input("Person full name: ").strip()
        if not name:
            break
        emp = input("  employee id (optional): ").strip() or None
        dept = input("  department  (optional): ").strip() or None
        res = fe.enroll_from_source(name, source=args.source, n_samples=args.samples,
                                    employee_id=emp, department=dept)
        print(f"  -> saved {res.saved} samples for {name}\n")
    print("Done. Run: python main.py train")


# ----------------------------------------------------------------- train -----
def cmd_train(args):
    engine = fe.FaceEngine()
    n, people = engine.train()
    _print_banner("Model trained")
    print(f"  samples: {n}")
    print(f"  people : {people}")
    print(f"  model  : {engine.recognizer is not None}")
    conn = db.get_conn()
    missing = db.unenrolled_people(conn)
    if missing:
        print("\n  NOTE: these people have no samples and were skipped:")
        for m in missing:
            print(f"    - {m['name']}")


# ------------------------------------------------------------------- run -----
def cmd_run(args):
    engine = fe.FaceEngine()
    if engine.recognizer is None:
        sys.exit("No trained model found. Run: python main.py train")

    conn = db.get_conn()
    tracker = fe.Tracker()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cam = fe.open_camera(args.source)
    show = not args.no_display and fe._can_show()
    last_proc = 0.0
    prev_gray = None
    start = time.time()
    frames = recognised = 0

    _print_banner("Live recognition + attendance (press Q to quit)")
    print(f"  shift {SHIFT_START}-{SHIFT_END}, late after {LATE_AFTER}")
    print(f"  display: {'on' if show else 'off (headless)'}   source: {args.source}")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frames += 1
            now = time.time()
            if args.interval and now - last_proc < args.interval:
                _maybe_imshow(show, "Attendance", frame)
                if _quit(show):
                    break
                continue
            last_proc = now

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = fe._maybe_scale(gray)
            sb = gray.shape[0] / small.shape[0]
            boxes = [(int(x * sb), int(y * sb), int(w * sb), int(h * sb))
                     for (x, y, w, h) in fe.detect_faces(small)]

            for box in boxes:
                x, y, w, h = box
                if w < fe.MIN_FACE_PX:
                    continue
                live = fe.liveness_score(prev_gray, gray, box)
                pid, conf = engine.recognise(gray, box)

                if pid is not None and fe.is_live(live):
                    name = engine.names.get(pid, "?")
                    if tracker.seen(pid, now):
                        snap = os.path.join(SNAPSHOT_DIR,
                                            f"{name}_{int(now)}.jpg")
                        cv2.imwrite(snap, frame)
                        effect = db.log_event(conn, pid, conf, snap,
                                              liveness=live, camera=str(args.source))
                        recognised += 1
                        _announce(name, effect)
                    colour, label = (0, 255, 0), f"{name} {conf:.0f}%"
                elif pid is not None:
                    colour, label = (0, 165, 255), f"{engine.names.get(pid,'?')} SPOOF?"
                else:
                    colour, label = (0, 0, 255), "UNKNOWN"
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            prev_gray = gray
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _maybe_imshow(show, "Attendance", frame)
            if _quit(show):
                break
            if args.seconds and time.time() - start > args.seconds:
                break
    finally:
        cam.release()
        if show:
            cv2.destroyAllWindows()

    db.mark_departures(conn)
    _print_banner("Session summary")
    print(f"  frames processed : {frames}")
    print(f"  attendance hits  : {recognised}")
    db.attendance_report(conn)


def _announce(name, effect):
    action = effect.get("action")
    if action == "punch_in":
        tag = " LATE" if effect.get("status") == "late" else ""
        print(f"  [IN ] {name}{tag}")
    elif action == "update":
        print(f"  [.. ] {name} (still present)")
    else:
        print(f"  [?  ] {name}")


def _maybe_imshow(show, title, frame):
    if show:
        cv2.imshow(title, frame)


def _quit(show):
    return show and (cv2.waitKey(1) & 0xFF == ord("q"))


# ---------------------------------------------------------------- report -----
def cmd_report(args):
    conn = db.get_conn()
    db.attendance_report(conn, args.date)
    if args.csv:
        db.export_csv(conn, date=args.date)


def cmd_people(args):
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT p.id, p.name, p.employee_id, p.department,"
        " COUNT(s.id) samples FROM people p"
        " LEFT JOIN face_samples s ON s.person_id=p.id GROUP BY p.id"
        " ORDER BY p.name").fetchall()
    _print_banner(f"Enrolled people ({len(rows)})")
    for r in rows:
        print(f"  #{r['id']:<3} {r['name']:<22}{(r['employee_id'] or '-'):<12}"
              f"{(r['department'] or '-'):<16}{r['samples']} samples")
    print("\nDB totals:", db.summary(conn))


# ------------------------------------------------------------------- API -----
def cmd_api(args):
    try:
        import uvicorn
    except ImportError:
        sys.exit("Install the API extras: pip install fastapi 'uvicorn[standard]'")
    from config import API_HOST, API_PORT
    print(f"Starting Face Recognition API on http://{API_HOST}:{API_PORT}")
    uvicorn.run("face_api:app", host=API_HOST, port=API_PORT, reload=False)


# -------------------------------------------------------------- selftest -----
def cmd_selftest(args):
    """Exercise the whole pipeline without a camera: synthetic face frames."""
    _print_banner("Self-test (no hardware required)")
    ok = True

    # 1. detection + quality + liveness on a synthetic image
    img = np.full((240, 240), 100, dtype=np.uint8)
    cv2.rectangle(img, (60, 60), (180, 180), 200, -1)  # bright blob
    cv2.circle(img, (95, 105), 8, 40, -1)              # eyes
    cv2.circle(img, (145, 105), 8, 40, -1)
    cv2.rectangle(img, (60, 60), (180, 180), 255, 2)
    q = fe.face_quality(img, (60, 60, 120, 120))
    prev = img.copy()
    moved = img.copy()
    cv2.circle(moved, (95, 105), 8, 60, -1)            # "blink"
    live = fe.liveness_score(prev, moved, (60, 60, 120, 120))
    print(f"  quality score    : {q:.3f}  (expect > 0)")
    print(f"  liveness score   : {live:.3f}  (movement detected)")

    # 2. full train/recognise round-trip on synthetic samples
    import tempfile
    tmp = tempfile.mkdtemp()
    conn = db.get_conn(os.path.join(tmp, "t.db"))
    pid = db.add_person(conn, "Selftest")
    rng = np.random.default_rng(0)
    for i in range(12):
        face = (rng.normal(128, 25, (fe.FACE_SIZE, fe.FACE_SIZE))
                .clip(0, 255).astype(np.uint8))
        p = os.path.join(tmp, f"{i}.jpg")
        cv2.imwrite(p, face)
        db.add_sample(conn, pid, p)

    import config
    engine = fe.FaceEngine(conn)
    engine.recognizer = fe._new_recognizer()
    faces, labels = [], []
    for r in conn.execute("SELECT person_id,image_path FROM face_samples"):
        im = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        faces.append(cv2.resize(im, (fe.FACE_SIZE, fe.FACE_SIZE)))
        labels.append(r["person_id"])
    engine.recognizer.train(faces, np.array(labels))
    pid2, conf = engine.recognise(faces[0], (0, 0, fe.FACE_SIZE, fe.FACE_SIZE))
    print(f"  train + predict  : person_id={pid2} conf={conf:.1f}%")
    print(f"  attendance fold  : {db.log_event(conn, pid, conf, None)}")
    db.attendance_report(conn)

    if q <= 0 or live is None or pid2 is None:
        ok = False
    _print_banner("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# ------------------------------------------------------------------ main -----
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("enroll", help="capture real samples for one person")
    e.add_argument("name")
    e.add_argument("--samples", type=int, default=40)
    e.add_argument("--source", default=CAMERA_SOURCE)
    e.add_argument("--employee-id", default=None)
    e.add_argument("--department", default=None)
    e.set_defaults(func=cmd_enroll)

    eb = sub.add_parser("enroll-batch", help="onboard several people in a row")
    eb.add_argument("--samples", type=int, default=40)
    eb.add_argument("--source", default=CAMERA_SOURCE)
    eb.set_defaults(func=cmd_enroll_batch)

    sub.add_parser("train", help="train the model on stored samples").set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="live recognition + attendance")
    r.add_argument("--source", default=CAMERA_SOURCE)
    r.add_argument("--no-display", action="store_true")
    r.add_argument("--seconds", type=float, default=0,
                   help="stop after N seconds (0 = run until Q)")
    r.add_argument("--interval", type=float, default=0,
                   help="seconds between processed frames")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="attendance report for a day")
    rep.add_argument("--date", default=None)
    rep.add_argument("--csv", action="store_true", help="also export CSV")
    rep.set_defaults(func=cmd_report)

    sub.add_parser("people", help="list enrolled people").set_defaults(func=cmd_people)
    sub.add_parser("api", help="run the REST API").set_defaults(func=cmd_api)

    st = sub.add_parser("selftest", help="verify the pipeline without a camera")
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
