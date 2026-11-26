from __future__ import annotations

import numpy as np

from redactor.detectors import Box, Detection


# Espande un riquadro di pad pixel senza uscire dai bordi
def expand_box(box: Box, pad: int, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad))


# Annerisce una copia dell'immagine: box espansi e, se c'è la maschera, tutto lo sfondo
def redact_image(
    image_rgb: np.ndarray,
    detections: list[Detection],
    subject_mask: np.ndarray | None,
    pad_px: int,
) -> tuple[np.ndarray, int]:
    output = image_rgb.copy()
    height, width = output.shape[:2]
    background_px = 0
    if subject_mask is not None:
        if subject_mask.shape != (height, width):
            raise ValueError("la maschera non ha le dimensioni dell'immagine")
        background = ~subject_mask
        output[background] = 0
        background_px = int(background.sum())
    for detection in detections:
        x1, y1, x2, y2 = expand_box(detection["box"], pad_px, width, height)
        output[y1:y2, x1:x2] = 0
    return output, background_px
