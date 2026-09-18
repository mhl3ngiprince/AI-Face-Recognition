# 5 · Medical Diagnosis Assistant - Real X-rays, Calibration, Audit & Ethics
A chest X-ray (pneumonia) decision-support system built the way clinical ML
should be: real image formats incl. **DICOM**, **probability calibration**, the
**medical metrics that matter** (sensitivity/specificity/AUROC), a full **audit
trail**, a **clinician feedback loop**, and a mandatory **safety disclaimer**.

## Real, measured results on the real dataset
The loader downloads the **real Kermany / Guangzhou chest X-ray set**
(5,856 images, CC-BY-4.0) straight from Hugging Face - the same data as the
popular Kaggle release, **no login required**. The offline
**HOG + calibrated logistic-regression** backend (chosen on evidence from
`benchmark_backends.py`) reaches, on a **properly split** 4,434 train /
782 validation / **624 held-out real test images**:

| Metric | Value |
|--------|-------|
| **AUROC** | **0.937** |
| **Sensitivity** | **98.97%** (386/390 pneumonia caught) |
| Specificity | 42.3% |
| Accuracy | 77.7% |
| Operating threshold | 0.467 (chosen on the 782-image validation set) |

### Why specificity is low - and why that is honest
This is a **screening** operating point: it deliberately catches ~99% of
pneumonia and accepts false alarms that a clinician clears. The threshold is a
choice, not a defect. `models/metrics.json` stores the full **operating curve**
so you can pick your own point. For example at threshold **0.95**:

| thr | sensitivity | specificity | accuracy |
|-----|-------------|-------------|----------|
| 0.47 | 0.99 | 0.42 | 0.78 |
| 0.75 | 0.98 | 0.50 | 0.80 |
| **0.95** | **0.96** | **0.56** | **0.825** |

### Backend selection (evidence, not guesswork)
`python benchmark_backends.py` trains and scores several real pipelines on the
real data. On the full set it found:

| backend | AUROC | fit time |
|---------|-------|----------|
| **logreg** | **0.937** | **1.2 s** |
| hist-grad-boost | 0.918 | 475 s |
| rbf-SVC (calibrated) | 0.913 | 20 |
| linear-SVC (calibrated) | 0.899 | 7 s |

So the default is logistic regression: best AUROC *and* fastest. Swap in a CNN
with `--backend cnn` once TensorFlow is installed.

> Warning:️ **Decision support only - not a diagnosis.** Every stored case carries the
> disclaimer. A qualified radiologist/clinician must review every result. This
> is not a regulated medical device.

## What's real here

| Concern | What this does |
|--------|-----------------|
| Images | JPEG/PNG **and DICOM (.dcm)** via `pydicom`, with modality rescale + MONOCHROME1 handling |
| Dataset | Real Kaggle `chest-xray-pneumonia` layout; auto-carves a validation split if absent |
| Models | **CNN** (Keras/TensorFlow) **or** **HOG + calibrated SVM** (scikit-learn) - training always works |
| Calibration | Isotonic / Platt scaling so a "70%" really is ~70% |
| Metrics | Sensitivity, specificity, precision, F1, **AUROC**, confusion matrix |
| Threshold | Chosen from validation (Youden **or** ≥90% sensitivity for screening), clamped sensibly |
| Audit | Every inference + every action is logged immutably (who/what/when) |
| Feedback loop | Clinicians record ground truth -> the report measures real-world agreement |
| Governance | Training metrics written to `models/metrics.json` |
| API | REST: predict, feedback, cases, report, audit |
| Tests | `pytest` covering imaging, calibration, metrics, DB, migration, dataset |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# Option A - train on the REAL chest X-ray dataset (no login needed):
python train.py --hf --backend classic   # downloads + trains + reports
python infer.py hf_cache_images/test/PNEUMONIA/test_00000.png --clinician dr_x
python report.py
# Option B - download the dataset to a folder first, then train:
python download_dataset.py --out chest_xray   # writes train/val/test/NORMAL|PNEUMONIA
python train.py --data chest_xray
# Option C - prove the pipeline fast with synthetic images (no download):
python sample_data.py sample_data
python train.py --data sample_data --backend classic
# Compare real backends on the real data and pick the best:
python benchmark_backends.py
# Option D - the Kaggle release instead (needs ~/.kaggle/kaggle.json):
python download_dataset.py --source kaggle
```

### DICOM support
```bash
pip install pydicom          # + python-gdcm or pylibjpeg for compressed DICOMs
python infer.py study.dcm --clinician dr_y --patient REF-77
```

### REST API
```bash
python -m uvicorn api:app --port 8005     # docs at /docs
```
| Method | Path | Purpose |
|-------|------|---------|
| GET  | `/health` | service + DB status |
| POST | `/predict` | upload an X-ray -> calibrated, audit-logged result |
| POST | `/feedback` | clinician ground truth |
| GET  | `/cases` · `/report` · `/audit` | governance views |

## How training works
1. **Discover** train/test (val carved from train if missing).
2. **Fit** the chosen backend.
3. **Calibrate** on validation (isotonic/Platt).
4. **Evaluate on the untouched test split** -> sensitivity/specificity/AUROC.
5. **Pick an operating threshold** from validation (Youden or ≥90% sens).
6. **Persist** model + calibrator + threshold + metrics.

## Interpreting the numbers (read this)
* **AUROC** measures ranking ability; **sensitivity/specificity** depend on the
  threshold you choose - always report both together.
* For a screening tool you usually want **high sensitivity**; use
  `--criterion sens90`.
* Never judge a model on accuracy alone with imbalanced data (pneumonia classes
  are not balanced).

## Governance & ethics
* The disclaimer is stored on **every** case, not just printed.
* The `audit` table records every inference, feedback and training run.
* `models/metrics.json` captures the exact performance of the deployed model.
* No patient identifiers are required; `patient_ref` is a free reference only -
  keep real identifiers out of this system or secure the DB accordingly.

## Tables
`cases` · `feedback` · `audit`

## Extensions
* Grad-CAM heatmaps so clinicians see *why*.
* External validation on local hospital data before any deployment.
* Regulatory pathway (e.g. CE/FDA) and human-factors testing.

## Web dashboard

Every project ships a server-rendered **web dashboard** (a real HTML page).

* **No emoji** ? all icons are inline **SVG** (defined in `../_shared/dashboard_kit.py`).
* **Real data only** ? every card is labelled with its data source, and the
  page reads the same SQLite tables the pipeline writes.
* **No CDN / no JavaScript required** ? charts are plain inline SVG.

Open it by running the project's API and visiting `/dashboard`:

    python <api-entrypoint>            # start the service
    # then open http://127.0.0.1:<port>/dashboard
