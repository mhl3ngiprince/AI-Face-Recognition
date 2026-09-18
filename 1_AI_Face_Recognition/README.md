# 1 · AI Face Recognition - Real Security & Attendance System

A real, deployable access-control / attendance system. **No simulation anywhere**:
it opens a real camera (webcam, video file or RTSP/CCTV stream), detects real
faces, checks they are *live* (anti-photo-spoof), recognises them with a trained
model and writes debounced attendance records to SQLite with snapshot evidence.

## Why this is production-grade, not a demo

| Concern | What this does |
|--------|-----------------|
| Video source | Webcam **or** recorded file **or** `rtsp://` IP camera - same code |
| Detection | Frontal **+ profile** cascades, merged with non-max suppression |
| Hardware quirks | Haar XMLs auto-downloaded & cached (OpenCV 5 no longer bundles them) |
| Image quality | Laplacian-sharpness + brightness + size gate rejects blurry captures |
| Anti-spoofing | Frame-to-frame motion check rejects printed photos / phone screens |
| False punches | A face must be confirmed over N frames; per-person cooldown de-bounces |
| Attendance policy | Automatic late / early-departure flags against shift times |
| Evidence | Snapshot saved for every committed event |
| Reporting | Human-readable report + CSV export for payroll/HR |
| Integration | Free REST API (enroll, train, recognize, attendance, export) |
| Tests | `pytest` suite covering detection, liveness, DB and training |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                 # edit camera + shift times if you like
python main.py selftest              # verify the whole pipeline with no camera
python main.py enroll "Alice"
python main.py train
python main.py run                   # live recognition + attendance
python main.py report --csv
```

### Point it at a real CCTV / IP camera
```bash
# .env
FR_CAMERA_SOURCE=rtsp://user:pass@192.168.1.50:554/stream1
```

### Zero-GUI / headless deployment (e.g. a gate box)
```bash
python main.py run --no-display --seconds 3600
```

## REST API
```bash
python main.py api                   # http://127.0.0.1:8001  (docs at /docs)
```
| Method | Path | Purpose |
|-------|------|---------|
| GET  | `/health` | service + model status |
| POST | `/people` | register a person |
| POST | `/enroll/{name}/sample` | enroll from an uploaded photo |
| POST | `/train` | retrain the model |
| POST | `/recognize` | recognise the largest face in a photo |
| GET  | `/attendance?date=YYYY-MM-DD` | attendance records |
| POST | `/attendance/export` | write the CSV |

## Commands
```
enroll <name>      enroll-batch      train       run
report [--date] [--csv]   people     api         selftest
```

## How recognition works
1. **Detect** faces (frontal + profile) on a downscaled grayscale frame.
2. **Quality-gate** each face; blurry/far faces are rejected during enrollment.
3. **Liveness**: compare the face region against the previous frame - a static
   photo produces ~0 motion and is rejected.
4. **Recognise** with LBPH (also supports `eigen`/`fisher` via `FR_RECOGNIZER`).
5. **Debounce**: confirm over `FR_CONFIRM_FRAMES` then cooldown `FR_EVENT_COOLDOWN`.
6. **Record** the event + snapshot, and fold it into today's attendance row.

## Tables
`people` · `face_samples` · `recognition_events` · `attendance`
(see `database.py`).

## Tuning
* Stricter/looser matching: `FR_CONFIDENCE_GATE` (lower = stricter).
* Bigger gates / more people: increase `FR_CONFIRM_FRAMES`.
* Faster on weak CPUs: lower `FR_PROCESS_SCALE` (e.g. `0.35`) and raise
  `FR_FRAME_INTERVAL`.

> Privacy: face data is personal data. Get consent, secure `faces.db`,
> snapshots and the `exports/` folder, and set a retention policy.

## Web dashboard

Every project ships a server-rendered **web dashboard** (a real HTML page).

* **No emoji** ? all icons are inline **SVG** (defined in `../_shared/dashboard_kit.py`).
* **Real data only** ? every card is labelled with its data source, and the
  page reads the same SQLite tables the pipeline writes.
* **No CDN / no JavaScript required** ? charts are plain inline SVG.

Open it by running the project's API and visiting `/dashboard`:

    python <api-entrypoint>            # start the service
    # then open http://127.0.0.1:<port>/dashboard
