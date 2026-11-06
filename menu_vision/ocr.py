"""OCR layer: image bytes -> OcrBox[].

Deliberately thin. The parser in `layout.py` never imports easyocr, so the
engine can be swapped (PaddleOCR, a vision LLM, ...) without touching parsing.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps

from .config import Settings
from .layout import parse_price
from .models import OcrBox


class OcrEngine(Protocol):
    name: str

    def read(self, image_bgr: np.ndarray) -> list[OcrBox]: ...


def load_image(image: bytes | str | Path) -> np.ndarray:
    """Decode an image to a BGR array, honouring EXIF orientation.

    Phone photos routinely carry a rotation flag. cv2.imread ignores it, so the
    whole menu comes back sideways and OCR falls apart. PIL + exif_transpose
    fixes that, which is why decoding goes through PIL rather than cv2.
    """
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
    else:
        raw = Path(image).read_bytes()
    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        array = np.asarray(rgb)
    return np.ascontiguousarray(array[:, :, ::-1])


def maybe_upscale(image_bgr: np.ndarray, min_width: int) -> tuple[np.ndarray, float]:
    """Upscale small images; easyocr misses thin strokes at low resolution."""
    height, width = image_bgr.shape[:2]
    if min_width <= 0 or width >= min_width:
        return image_bgr, 1.0
    scale = min_width / width
    resized = Image.fromarray(image_bgr[:, :, ::-1]).resize(
        (int(width * scale), int(height * scale)), Image.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(resized)[:, :, ::-1]), scale


class EasyOcrEngine:
    """easyocr wrapper.

    The Reader loads ~100 MB of weights on construction (seconds), so it is
    built once per process and reused. easyocr's reader is not documented as
    thread-safe, hence the lock: under the future FastAPI server concurrent
    requests will serialise on OCR rather than corrupt each other.
    """

    name = "easyocr"

    def __init__(self, langs: Iterable[str] = ("ch_sim", "en"), gpu: bool = False) -> None:
        self._langs = list(langs)
        self._gpu = gpu
        self._reader = None
        self._lock = threading.Lock()

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self._langs, gpu=self._gpu, verbose=False)
        return self._reader

    def read(self, image_bgr: np.ndarray) -> list[OcrBox]:
        reader = self._get_reader()
        with self._lock:
            detections = reader.readtext(image_bgr)
        boxes: list[OcrBox] = []
        for quad, text, score in detections:
            xs = [point[0] for point in quad]
            ys = [point[1] for point in quad]
            cleaned = text.strip()
            if not cleaned:
                continue
            boxes.append(
                OcrBox(
                    text=cleaned,
                    confidence=float(score),
                    x0=int(min(xs)),
                    y0=int(min(ys)),
                    x1=int(max(xs)),
                    y1=int(max(ys)),
                )
            )
        return boxes


_ENGINES: dict[tuple, EasyOcrEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(settings: Settings) -> OcrEngine:
    """Process-wide cached engine (one weight load for the whole server)."""
    key = (tuple(settings.ocr_langs), settings.ocr_gpu)
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = EasyOcrEngine(settings.ocr_langs, settings.ocr_gpu)
            _ENGINES[key] = engine
    return engine


def extract_boxes(image: bytes | str | Path, settings: Settings, engine: OcrEngine | None = None) -> tuple[list[OcrBox], tuple[int, int], float]:
    """Decode, upscale if needed, run OCR, and drop low-confidence noise."""
    image_bgr = load_image(image)
    original_height, original_width = image_bgr.shape[:2]
    image_bgr, scale = maybe_upscale(image_bgr, settings.upscale_below_width)
    engine = engine or get_engine(settings)
    boxes = engine.read(image_bgr)
    if scale != 1.0:
        # Scale coordinates back so they map onto the original image, which is
        # what the debug overlay and any bounding-box UI will draw on.
        inv = 1.0 / scale
        for box in boxes:
            box.x0 = int(box.x0 * inv)
            box.x1 = int(box.x1 * inv)
            box.y0 = int(box.y0 * inv)
            box.y1 = int(box.y1 * inv)
    kept = [
        box
        for box in boxes
        if box.confidence >= settings.min_confidence
        # Rescue a blurry price: low score, but unmistakably a price string.
        or (box.confidence >= settings.price_conf_floor and parse_price(box.text) is not None)
    ]
    return kept, (original_width, original_height), scale
