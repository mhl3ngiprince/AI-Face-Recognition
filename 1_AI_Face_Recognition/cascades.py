"""Robust face-detector provisioning.

OpenCV 5.x no longer ships the classic Haar cascade XML files inside
``cv2.data.haarcascades`` (the folder is there but empty). Rather than
crash, this module:

1. looks for the XML locally (``models/`` next to this file),
2. otherwise downloads it **once** from the official OpenCV repository and
   caches it,
3. exposes a cached ``get_detector(name)``.

If the machine is offline and the file is absent we raise a clear, actionable
error instead of letting OpenCV print a cryptic persistence message.
"""
from __future__ import annotations

import os
import urllib.request

import cv2

_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_cache: dict[str, "cv2.CascadeClassifier"] = {}


def _search_paths(filename: str) -> list[str]:
    """Every place a cascade XML might already live on this machine."""
    cands = [os.path.join(_LOCAL_DIR, filename)]
    try:
        cands.append(os.path.join(cv2.data.haarcascades, filename))
    except Exception:
        pass
    return cands


def ensure_cascade(filename: str) -> str:
    """Return a local path to `filename`, downloading it if necessary."""
    for path in _search_paths(filename):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path

    os.makedirs(_LOCAL_DIR, exist_ok=True)
    dest = os.path.join(_LOCAL_DIR, filename)
    url = _BASE + filename
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
    except Exception as exc:  # network/offline
        if os.path.isfile(dest):
            return dest
        raise RuntimeError(
            f"Face cascade '{filename}' is not bundled with your OpenCV build "
            f"and could not be downloaded ({exc}).\n"
            f"Fix it in one of two ways:\n"
            f"  * get online once so it caches to {dest}, or\n"
            f"  * download {url} and drop it in {_LOCAL_DIR}."
        ) from exc
    return dest


def get_detector(filename: str = "haarcascade_frontalface_default.xml"):
    """Cached CascadeClassifier loader."""
    if filename not in _cache:
        cascade = cv2.CascadeClassifier(ensure_cascade(filename))
        if cascade.empty():
            raise RuntimeError(f"Loaded '{filename}' but OpenCV reports it empty.")
        _cache[filename] = cascade
    return _cache[filename]
