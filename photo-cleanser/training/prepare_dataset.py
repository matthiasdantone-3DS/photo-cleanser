from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_REPORTED_PROBLEMS = 20
DEFAULT_DEST = Path(__file__).resolve().parent / "dataset"

logger = logging.getLogger("prepare_dataset")


# Elenca le immagini presenti in una cartella (ricorsivo)
def list_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# Trova il file di etichette di un'immagine: accanto ad essa o in una cartella labels parallela
def find_label(image: Path, source: Path) -> Path | None:
    candidates = [image.with_suffix(".txt")]
    parts = list(image.relative_to(source).parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        swapped = parts[:index] + ["labels"] + parts[index + 1:]
        candidates.append(source.joinpath(*swapped).with_suffix(".txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# Controlla un file di etichette YOLO (classe cx cy w h, tutto normalizzato) e restituisce i problemi
def check_label(path: Path, class_count: int) -> list[str]:
    problems: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path.name}: file illeggibile ({exc})"]
    for number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            problems.append(
                f"{path.name}:{number}: servono 5 valori (classe cx cy w h), trovati {len(fields)}"
            )
            continue
        try:
            class_id = int(fields[0])
            cx, cy, w, h = (float(value) for value in fields[1:])
        except ValueError:
            problems.append(f"{path.name}:{number}: valori non numerici")
            continue
        if not 0 <= class_id < class_count:
            problems.append(f"{path.name}:{number}: classe {class_id} fuori dall'elenco ({class_count} classi)")
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)) or w <= 0 or h <= 0:
            problems.append(f"{path.name}:{number}: coordinate non normalizzate tra 0 e 1")
    return problems


# Legge i nomi delle classi: da --classes, da classes.txt nella sorgente, oppure "logo"
def resolve_classes(arg_classes: list[str] | None, source: Path) -> list[str]:
    if arg_classes:
        return arg_classes
    classes_file = source / "classes.txt"
    if classes_file.is_file():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name]
        if names:
            return names
    return ["logo"]


# Scrive data.yaml nel formato atteso da ultralytics
def write_data_yaml(dest: Path, classes: list[str]) -> Path:
    lines = [
        f"path: {dest.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {number}: {json.dumps(name, ensure_ascii=False)}" for number, name in enumerate(classes)]
    target = dest / "data.yaml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# Copia le coppie immagine/etichetta nelle cartelle train e val; restituisce (train, val, senza_etichetta)
def build_dataset(
    pairs: list[tuple[Path, Path | None]], dest: Path, val_ratio: float, seed: int
) -> tuple[int, int, int]:
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    val_count = min(len(pairs) - 1, max(1, round(len(pairs) * val_ratio)))
    val_positions = set(order[:val_count])
    for split in ("train", "val"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)
    unlabeled = 0
    for position, (image, label) in enumerate(pairs):
        split = "val" if position in val_positions else "train"
        stem = f"{position:05d}_{image.stem}"
        shutil.copy2(image, dest / "images" / split / f"{stem}{image.suffix.lower()}")
        target_label = dest / "labels" / split / f"{stem}.txt"
        if label is None:
            target_label.write_text("", encoding="utf-8")
            unlabeled += 1
        else:
            shutil.copy2(label, target_label)
    return len(pairs) - val_count, val_count, unlabeled


# Controlla il dataset YOLO, lo divide in train/val e scrive data.yaml
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepara un dataset YOLO per l'allenamento")
    parser.add_argument("--source", required=True, type=Path, help="cartella con immagini ed etichette .txt")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="cartella del dataset pronto")
    parser.add_argument("--classes", nargs="+", default=None, help="nomi delle classi (default: classes.txt o 'logo')")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="quota di immagini per la validazione")
    parser.add_argument("--seed", type=int, default=42, help="seme della divisione casuale")
    parser.add_argument("--force", action="store_true", help="sostituisce un dataset già presente in dest")
    args = parser.parse_args(argv)

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio deve essere tra 0 e 1 (esclusi)")
    if not args.source.is_dir():
        logger.error("cartella sorgente non trovata: %s", args.source)
        return 1
    classes = resolve_classes(args.classes, args.source)
    images = list_images(args.source)
    if len(images) < 2:
        logger.error("servono almeno 2 immagini, trovate %d", len(images))
        return 1

    pairs = [(image, find_label(image, args.source)) for image in images]
    problems: list[str] = []
    for _, label in pairs:
        if label is not None:
            problems.extend(check_label(label, len(classes)))
    if problems:
        for problem in problems[:MAX_REPORTED_PROBLEMS]:
            logger.error(problem)
        if len(problems) > MAX_REPORTED_PROBLEMS:
            logger.error("... e altri %d problemi", len(problems) - MAX_REPORTED_PROBLEMS)
        logger.error("correggi le etichette e rilancia: nessun file è stato copiato")
        return 1

    if (args.dest / "images").exists():
        if not args.force:
            logger.error("%s contiene già un dataset: usa --force per sostituirlo", args.dest)
            return 1
        shutil.rmtree(args.dest / "images")
        shutil.rmtree(args.dest / "labels", ignore_errors=True)

    train_count, val_count, unlabeled = build_dataset(pairs, args.dest, args.val_ratio, args.seed)
    data_yaml = write_data_yaml(args.dest, classes)
    logger.info("classi: %s", ", ".join(classes))
    logger.info("immagini: %d per l'allenamento, %d per la validazione", train_count, val_count)
    if unlabeled:
        logger.info("%d immagini senza etichette: usate come esempi negativi (nessun logo)", unlabeled)
    logger.info("dataset pronto: %s", data_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
