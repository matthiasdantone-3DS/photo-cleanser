from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from redactor.segmentation import SubjectSegmenter


# Calcola con l'algoritmo Python originale (MediaPipe) la maschera di riferimento di un'immagine, in forma compressa
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maschera di riferimento della pipeline Python")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    image = np.array(Image.open(args.image).convert("RGB"))
    result = SubjectSegmenter(Settings()).segment(image)
    flat = result.mask.astype(np.uint8).reshape(-1)
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    args.out.write_text(
        json.dumps({"width": image.shape[1], "height": image.shape[0], "first": int(flat[0]), "runs": runs}),
        encoding="utf-8",
    )
    print("maschera scritta:", args.out, "soggetto", int(flat.sum()), "px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
