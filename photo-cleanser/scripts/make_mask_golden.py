from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from redactor.segmentation import build_subject_mask


# Maschera casuale a macchie: rumore sfocato e poi binarizzato
def random_mask(rng: np.random.Generator, width: int, height: int, blur: int, density: float) -> np.ndarray:
    noise = rng.random((height, width)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), blur)
    return (smooth > np.quantile(smooth, 1.0 - density)).astype(np.uint8)


# Genera maschere casuali e i risultati di riferimento ottenuti con la pipeline Python/OpenCV
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dati di riferimento per le operazioni sulle maschere")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(11)
    settings = Settings()
    cases = []
    for number, (w, h, blur, density) in enumerate(
        [(64, 48, 2, 0.35), (97, 61, 1, 0.5), (200, 150, 3, 0.25), (131, 90, 1, 0.12), (50, 50, 4, 0.6), (180, 120, 2, 0.45)]
    ):
        for repeat in range(3):
            raw = random_mask(rng, w, h, blur, density)
            probabilities = raw.astype(np.float32) * 0.9 + 0.05
            result = build_subject_mask(probabilities, settings)
            cases.append(
                {
                    "w": w,
                    "h": h,
                    "input": "".join(map(str, raw.reshape(-1))),
                    "empty": bool(result.empty),
                    "expected": "".join(map(str, result.mask.astype(np.uint8).reshape(-1))),
                }
            )
    cases.append(
        {
            "w": 20, "h": 10, "input": "0" * 200, "empty": True, "expected": "0" * 200,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print("casi di riferimento:", len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
