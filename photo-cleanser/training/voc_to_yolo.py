from __future__ import annotations

import argparse
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

logger = logging.getLogger("voc_to_yolo")

# Riquadro normalizzato: nome classe, centro x/y, larghezza, altezza
YoloBox = tuple[str, float, float, float, float]


# Indicizza le immagini della cartella sorgente per nome (senza estensione)
def index_images(source: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, []).append(path)
    return index


# Trova l'immagine di un file XML: prima accanto ad esso, poi in tutta la cartella
def find_image(xml_path: Path, index: dict[str, list[Path]]) -> Path | None:
    candidates = index.get(xml_path.stem, [])
    for candidate in candidates:
        if candidate.parent == xml_path.parent:
            return candidate
    return candidates[0] if candidates else None


# Legge un file Pascal VOC e restituisce i riquadri normalizzati in formato YOLO
def read_voc(xml_path: Path, image_path: Path) -> list[YoloBox]:
    root = ET.parse(xml_path).getroot()
    width = height = 0
    size = root.find("size")
    if size is not None:
        try:
            width = int(float(size.findtext("width", "0")))
            height = int(float(size.findtext("height", "0")))
        except ValueError:
            width = height = 0
    if width <= 0 or height <= 0:
        with Image.open(image_path) as image:
            width, height = image.size
    boxes: list[YoloBox] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        if not name or bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", ""))
            ymin = float(bndbox.findtext("ymin", ""))
            xmax = float(bndbox.findtext("xmax", ""))
            ymax = float(bndbox.findtext("ymax", ""))
        except ValueError:
            continue
        xmin, xmax = sorted((min(max(xmin, 0.0), width), min(max(xmax, 0.0), width)))
        ymin, ymax = sorted((min(max(ymin, 0.0), height), min(max(ymax, 0.0), height)))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            (
                name,
                (xmin + xmax) / 2 / width,
                (ymin + ymax) / 2 / height,
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return boxes


# Converte tutte le annotazioni trovate e scrive immagini ed etichette in dest
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Converte annotazioni Pascal VOC (XML, es. OpenLogo, LogoDet-3K) in etichette YOLO"
    )
    parser.add_argument("--source", required=True, type=Path, help="cartella con immagini e file XML")
    parser.add_argument("--dest", required=True, type=Path, help="cartella di destinazione")
    parser.add_argument("--class-name", default="logo", help="nome dell'unica classe (default: logo)")
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="mantiene le classi originali (un marchio per classe) invece di unirle in una sola",
    )
    args = parser.parse_args(argv)

    xml_files = sorted(args.source.rglob("*.xml"))
    if not xml_files:
        logger.error("nessun file XML trovato in %s", args.source)
        return 1
    index = index_images(args.source)

    converted: list[tuple[Path, list[YoloBox]]] = []
    skipped = 0
    for xml_path in xml_files:
        image_path = find_image(xml_path, index)
        if image_path is None:
            logger.warning("immagine non trovata per %s", xml_path.name)
            skipped += 1
            continue
        try:
            converted.append((image_path, read_voc(xml_path, image_path)))
        except (ET.ParseError, OSError) as exc:
            logger.warning("%s ignorato: %s", xml_path.name, exc)
            skipped += 1

    if args.keep_classes:
        names = sorted({box[0] for _, boxes in converted for box in boxes})
    else:
        names = [args.class_name]
    class_ids = {name: number for number, name in enumerate(names)}

    args.dest.mkdir(parents=True, exist_ok=True)
    for position, (image_path, boxes) in enumerate(converted):
        stem = f"{position:05d}_{image_path.stem}"
        shutil.copy2(image_path, args.dest / f"{stem}{image_path.suffix.lower()}")
        lines = []
        for name, cx, cy, w, h in boxes:
            class_id = class_ids[name] if args.keep_classes else 0
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (args.dest / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (args.dest / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    logger.info("convertite %d immagini (%d ignorate), classi: %s", len(converted), skipped, ", ".join(names))
    logger.info("passo successivo: python training/prepare_dataset.py --source %s", args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
