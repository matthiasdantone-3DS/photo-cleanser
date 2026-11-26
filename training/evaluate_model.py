from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Rende importabile config.py anche lanciando lo script da training/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LOGO_PATH

RUNS_DIR = Path(__file__).resolve().parent / "runs"

logger = logging.getLogger("evaluate_model")


# Misura il modello su un dataset e/o salva le immagini con i riquadri trovati
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Valuta un modello di loghi")
    parser.add_argument("--model", type=Path, default=DEFAULT_LOGO_PATH, help="modello .pt (default: models/logo.pt)")
    parser.add_argument("--data", type=Path, default=None, help="data.yaml per calcolare precisione e richiamo")
    parser.add_argument("--preview", type=Path, default=None, help="cartella di foto di prova da annotare")
    parser.add_argument("--conf", type=float, default=0.4, help="soglia di confidenza per l'anteprima")
    parser.add_argument("--imgsz", type=int, default=640, help="dimensione in pixel")
    parser.add_argument("--device", default=None, help="cpu, 0 per la prima GPU (default: automatico)")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        logger.error("modello non trovato: %s", args.model)
        return 1
    if args.data is None and args.preview is None:
        logger.error("indica almeno --data oppure --preview")
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if args.data is not None:
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
        logger.info(
            "precisione %.3f, richiamo %.3f, mAP50 %.3f, mAP50-95 %.3f",
            metrics.box.mp,
            metrics.box.mr,
            metrics.box.map50,
            metrics.box.map,
        )
    if args.preview is not None:
        model.predict(
            source=str(args.preview),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=True,
            project=str(RUNS_DIR),
            name="preview",
            exist_ok=True,
        )
        logger.info("immagini annotate salvate in %s", RUNS_DIR / "preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
