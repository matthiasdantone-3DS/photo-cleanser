from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Rende importabile config.py anche lanciando lo script da training/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEFAULT_LOGO_PATH, LOGO_KEY, sha256_file, write_lock_checksum

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_DATA = Path(__file__).resolve().parent / "dataset" / "data.yaml"

logger = logging.getLogger("train_logo")


# Copia il modello migliore in models/logo.pt (con copia di sicurezza) e aggiorna il lock
def install_model(best: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(f"{target.stem}.{datetime.now():%Y%m%d-%H%M%S}.pt")
        shutil.copy2(target, backup)
        logger.info("modello precedente salvato in %s", backup)
    shutil.copy2(best, target)
    checksum = sha256_file(target)
    write_lock_checksum(
        LOGO_KEY, checksum, "compilato da training/train_logo.py dopo l'allenamento"
    )
    return checksum


# Allena un modello YOLO sui loghi, lo valuta e (di default) lo installa per il sito e la CLI
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Allena il rilevatore di loghi (YOLO)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data.yaml creato da prepare_dataset.py")
    parser.add_argument("--base-model", default="yolov8n.pt", help="pesi di partenza (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="numero di epoche")
    parser.add_argument("--imgsz", type=int, default=640, help="dimensione di allenamento in pixel")
    parser.add_argument("--batch", type=int, default=8, help="immagini per passo (abbassa se finisce la memoria)")
    parser.add_argument("--patience", type=int, default=20, help="epoche senza miglioramenti prima di fermarsi")
    parser.add_argument("--device", default=None, help="cpu, 0 per la prima GPU (default: automatico)")
    parser.add_argument("--name", default="logo", help="nome della cartella dei risultati")
    parser.add_argument("--no-install", action="store_true", help="non copiare il modello in models/logo.pt")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        logger.error("data.yaml non trovato: %s (esegui prima training/prepare_dataset.py)", args.data)
        return 1

    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    # Disattiva l'invio di dati di utilizzo di ultralytics
    try:
        ultralytics_settings.update(sync=False)
    except Exception as exc:
        logger.warning("impossibile disattivare la sincronizzazione ultralytics: %s", exc)

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=False,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        logger.error("allenamento terminato ma %s non esiste", best)
        return 1

    metrics = YOLO(str(best)).val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    logger.info(
        "risultati sulla validazione: precisione %.3f, richiamo %.3f, mAP50 %.3f, mAP50-95 %.3f",
        metrics.box.mp,
        metrics.box.mr,
        metrics.box.map50,
        metrics.box.map,
    )
    logger.info("modello migliore: %s", best)

    if args.no_install:
        return 0
    checksum = install_model(best, DEFAULT_LOGO_PATH)
    logger.info("installato in %s (sha256 %s)", DEFAULT_LOGO_PATH, checksum)
    logger.info("riavvia il sito (python serve.py): la voce 'Loghi' in alto diventa attiva")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
