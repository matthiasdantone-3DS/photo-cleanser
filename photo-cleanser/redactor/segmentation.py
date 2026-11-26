from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from config import Settings

logger = logging.getLogger(__name__)


# Esito della segmentazione: maschera booleana del soggetto e flag di maschera vuota
@dataclass
class SubjectMask:
    mask: np.ndarray
    empty: bool


# Interfaccia comune dei segmentatori (utile anche per i mock nei test)
class Segmenter(Protocol):
    def segment(self, image_rgb: np.ndarray) -> SubjectMask: ...


# Rimuove le componenti connesse piccole rispetto alla più grande
def remove_small_components(binary: np.ndarray, min_ratio: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, dtype=bool)
    keep[1:] = areas >= areas.max() * min_ratio
    return keep[labels].astype(np.uint8)


# Binarizza la maschera di probabilità, chiude i buchi piccoli e filtra le componenti
def build_subject_mask(probabilities: np.ndarray, settings: Settings) -> SubjectMask:
    probabilities = probabilities.reshape(probabilities.shape[:2])
    binary = (probabilities >= settings.mask_threshold).astype(np.uint8)
    if not binary.any():
        return SubjectMask(mask=np.zeros(binary.shape, dtype=bool), empty=True)
    kernel = np.ones((settings.morph_kernel, settings.morph_kernel), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = remove_small_components(closed, settings.mask_min_component_ratio)
    return SubjectMask(mask=cleaned.astype(bool), empty=False)


# Dice se il centro di un riquadro cade dentro la maschera del soggetto
def center_inside_mask(box: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    height, width = mask.shape
    cx = min(max((box[0] + box[2]) // 2, 0), width - 1)
    cy = min(max((box[1] + box[3]) // 2, 0), height - 1)
    return bool(mask[cy, cx])


# Segmentazione persona con MediaPipe Selfie Segmentation (modello incluso nel pacchetto)
class SubjectSegmenter:
    def __init__(self, settings: Settings) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "questa versione di mediapipe non include più mp.solutions: "
                "installa mediapipe>=0.10.9,<0.10.30 (vedi requirements.txt)"
            )
        self._settings = settings
        self._model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    # Produce la maschera del soggetto principale per un'immagine RGB uint8
    def segment(self, image_rgb: np.ndarray) -> SubjectMask:
        result = self._model.process(np.ascontiguousarray(image_rgb))
        if result.segmentation_mask is None:
            raise RuntimeError("MediaPipe non ha prodotto alcuna maschera")
        return build_subject_mask(result.segmentation_mask, self._settings)

    # Libera le risorse del modello
    def close(self) -> None:
        self._model.close()
