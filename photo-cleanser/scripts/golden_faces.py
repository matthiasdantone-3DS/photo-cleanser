from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


# Esegue YuNet con OpenCV (l'algoritmo originale) su un'immagine e salva i volti trovati, per il confronto con il browser
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Volti di riferimento calcolati con OpenCV")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score", type=float, default=0.6)
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.image))
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(args.model), "", (width, height), args.score, 0.3, 5000)
    _, faces = detector.detect(image)
    rows = [] if faces is None else [[float(v) for v in row] for row in faces]
    args.out.write_text(json.dumps({"width": width, "height": height, "faces": rows}), encoding="utf-8")
    print("volti di riferimento:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
