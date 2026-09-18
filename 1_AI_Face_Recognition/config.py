"""Central configuration - set via .env or environment variables."""
import os
from dotenv import load_dotenv
load_dotenv()


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------- storage ---
DB_PATH = os.getenv("FR_DB_PATH", "faces.db")
MODEL_PATH = os.getenv("FR_MODEL_PATH", "lbph_model.yml")
DATASET_DIR = os.getenv("FR_DATASET_DIR", "faces_dataset")
SNAPSHOT_DIR = os.getenv("FR_SNAPSHOT_DIR", "snapshots")
EXPORT_DIR = os.getenv("FR_EXPORT_DIR", "exports")

# ---------------------------------------------------------------- camera ----
# CAMERA_SOURCE accepts:
#   * an integer index -> local webcam  (e.g. "0")
#   * a video file path -> recorded footage (e.g. "gate.mp4")
#   * an RTSP/HTTP URL -> IP camera / CCTV  (e.g. "rtsp://...")
CAMERA_SOURCE = os.getenv("FR_CAMERA_SOURCE", os.getenv("FR_CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("FR_CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("FR_CAMERA_HEIGHT", "720"))
FRAME_INTERVAL = float(os.getenv("FR_FRAME_INTERVAL", "0"))
PROCESS_SCALE = float(os.getenv("FR_PROCESS_SCALE", "0.5"))

# ------------------------------------------------------------- recognition --
# LBPH returns an L2 distance: LOWER is a *better* match.
CONFIDENCE_GATE = float(os.getenv("FR_CONFIDENCE_GATE", "80"))
RECOGNIZER = os.getenv("FR_RECOGNIZER", "lbph").lower()   # lbph | eigen | fisher
MIN_FACE_PX = int(os.getenv("FR_MIN_FACE_PX", "80"))
FACE_SIZE = int(os.getenv("FR_FACE_SIZE", "200"))
# A person must be seen this many times before attendance is committed,
# which stops one blurry frame creating a false punch-in.
CONFIRM_FRAMES = int(os.getenv("FR_CONFIRM_FRAMES", "3"))
# Seconds before the same person can be logged again (de-bounce).
EVENT_COOLDOWN = float(os.getenv("FR_EVENT_COOLDOWN", "30"))

# ------------------------------------------------------------ anti-spoof ----
REQUIRE_LIVENESS = _bool("FR_REQUIRE_LIVENESS", "true")
LIVENESS_EPS = float(os.getenv("FR_LIVENESS_EPS", "2.5"))

# ------------------------------------------------------------- attendance ---
SHIFT_START = os.getenv("FR_SHIFT_START", "08:00")
LATE_AFTER = os.getenv("FR_LATE_AFTER", "09:00")
SHIFT_END = os.getenv("FR_SHIFT_END", "17:00")
TIMEZONE = os.getenv("FR_TIMEZONE", "")

# ------------------------------------------------------------------- API ----
API_HOST = os.getenv("FR_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("FR_API_PORT", "8001"))

ATTENDANCE_TABLE = "attendance"
