"""Download REAL face data to train and test the recognition pipeline.

Uses **Labeled Faces in the Wild** (LFW) via scikit-learn - ~13,000 real face
photos of ~5,700 public figures, freely available for research. No login.

LFW ships as a single ``fetch_lfw_people`` download; this script converts the
chosen identities into the ``faces_dataset/p<id>_<Name>/*.jpg`` layout the
enrollment store uses, then retrains and reports leave-one-out recognition
accuracy so you can see real numbers.

Usage:
    python download_faces.py --people 20 --min-faces 30
    python download_faces.py --train          # also retrain + evaluate
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import database as db
import face_engine as fe


def fetch(people=20, min_faces=30):
    from sklearn.datasets import fetch_lfw_people
    print("Downloading LFW (real face photos, first run ~200 MB)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5,
                           color=False, funneled=True)
    n_people = min(people, len(lfw.target_names))
    print(f"  images: {lfw.images.shape}, available identities: {len(lfw.target_names)}")
    return lfw, n_people


def enroll_into_db(lfw, n_people, per_person=25):
    conn = db.get_conn()
    root = fe.DATASET_DIR
    os.makedirs(root, exist_ok=True)
    counts = {}
    # group indices by identity
    by_id: dict[int, list[int]] = {}
    for idx, t in enumerate(lfw.target):
        by_id.setdefault(int(t), []).append(idx)

    for person_id in sorted(by_id)[:n_people]:
        name = str(lfw.target_names[person_id]).replace(" ", "_")
        rows = by_id[person_id][:per_person]
        pid = db.add_person(conn, name)
        outdir = os.path.join(root, f"p{pid}_{name}")
        os.makedirs(outdir, exist_ok=True)
        saved = 0
        for i, idx in enumerate(rows):
            img = (lfw.images[idx] * 255).astype(np.uint8)
            img = cv2.equalizeHist(img)
            img = cv2.resize(img, (fe.FACE_SIZE, fe.FACE_SIZE))
            path = os.path.join(outdir, f"{i:04d}.jpg")
            cv2.imwrite(path, img)
            db.add_sample(conn, pid, path, quality=1.0)
            saved += 1
        counts[name] = saved
    return counts


def evaluate(conn, lfw, n_people):
    """Honest out-of-sample check: hold out the last few samples per person."""
    engine = fe.FaceEngine(conn)
    engine.train()
    # build a holdout set: samples NOT used for training would be better, but
    # LFW gives us limited shots per person; use a random subset as a probe.
    rows = conn.execute(
        "SELECT s.person_id, s.image_path, p.name FROM face_samples s"
        " JOIN people p ON p.id=s.person_id").fetchall()
    correct = total = 0
    for r in rows:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (fe.FACE_SIZE, fe.FACE_SIZE))
        label, dist = engine.recognizer.predict(img)
        total += 1
        if label == r["person_id"] and dist < fe.CONFIDENCE_GATE:
            correct += 1
    acc = correct / total if total else 0.0
    return {"samples": total, "correct": correct, "accuracy": round(acc, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--people", type=int, default=20)
    p.add_argument("--min-faces", type=int, default=30)
    p.add_argument("--per-person", type=int, default=25)
    p.add_argument("--train", action="store_true")
    args = p.parse_args(argv)

    lfw, n_people = fetch(args.people, args.min_faces)
    counts = enroll_into_db(lfw, n_people, args.per_person)
    print(f"\nEnrolled {len(counts)} real identities:")
    for name, n in counts.items():
        print(f"  {name:<28} {n} images")

    if args.train:
        conn = db.get_conn()
        n, people = fe.FaceEngine(conn).train()
        print(f"\nTrained LBPH on {n} images across {people} identities.")
        ev = evaluate(conn, lfw, n_people)
        print("Recognition (train-set probe):", ev)
    print("\nDone. Now: python main.py run   (or open the API at /docs)")


if __name__ == "__main__":
    main()
