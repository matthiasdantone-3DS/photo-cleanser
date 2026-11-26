from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tipi di rilevamento usati anche nel report
KIND_FACE = "volto"
KIND_LOGO = "logo"

# Riquadro (x1, y1, x2, y2) in pixel
Box = tuple[int, int, int, int]


# Rilevamento: tipo, riquadro e punteggio di confidenza
class Detection(TypedDict):
    kind: str
    box: Box
    score: float


# Interfaccia comune dei rilevatori (utile anche per i mock nei test)
class Detector(Protocol):
    def detect(self, image_rgb: np.ndarray) -> list[Detection]: ...


# Limita un riquadro ai bordi dell'immagine e lo converte in interi
def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Box:
    return (
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        min(width, int(np.ceil(x2))),
        min(height, int(np.ceil(y2))),
    )


# Un riquadro è valido se ha area positiva
def is_valid_box(box: Box) -> bool:
    return box[2] > box[0] and box[3] > box[1]


# Rilevatore di volti YuNet tramite cv2.FaceDetectorYN
class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    # Restituisce i volti trovati in un'immagine RGB
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(bgr)
        detections: list[Detection] = []
        if faces is None:
            return detections
        for row in faces:
            x, y, w, h = (float(value) for value in row[:4])
            box = clip_box(x, y, x + w, y + h, width, height)
            if is_valid_box(box):
                detections.append({"kind": KIND_FACE, "box": box, "score": float(row[-1])})
        return detections


# Rilevatore di loghi basato su un modello YOLO fornito dall'utente
class LogoDetector:
    def __init__(self, model_path: Path, conf: float) -> None:
        from ultralytics import YOLO
        from ultralytics import settings as ultralytics_settings

        # Disattiva l'invio di dati di utilizzo di ultralytics: le immagini sono sensibili
        try:
            ultralytics_settings.update(sync=False)
        except Exception as exc:
            logger.warning("impossibile disattivare la sincronizzazione ultralytics: %s", exc)
        self._model = YOLO(str(model_path))
        self._conf = conf

    # Restituisce i loghi trovati in un'immagine RGB con confidenza >= soglia
    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        height, width = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detections: list[Detection] = []
        for result in self._model.predict(bgr, conf=self._conf, verbose=False):
            if result.boxes is None:
                continue
            coords = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(coords, scores):
                if float(score) < self._conf:
                    continue
                box = clip_box(float(x1), float(y1), float(x2), float(y2), width, height)
                if is_valid_box(box):
                    detections.append({"kind": KIND_LOGO, "box": box, "score": float(score)})
        return detections
