from __future__ import annotations

import io
import logging
import struct
from pathlib import Path

from PIL import Image, ImageOps, IptcImagePlugin, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Firma iniziale di ogni file PNG
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Unici chunk ammessi in un PNG pulito: struttura e pixel
ALLOWED_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}

# Formati di ingresso accettati
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Chiavi di Pillow (img.info) associate a ciascun tipo di metadato
EXIF_KEYS = {"exif", "Raw profile type exif"}
XMP_KEYS = {"xmp", "XML:com.adobe.xmp"}
IPTC_KEYS = {"photoshop", "Raw profile type iptc"}


# Errore esplicito quando nel file restano metadati
class MetadataError(RuntimeError):
    pass


# Aggiunge un tipo di metadato alla lista senza duplicati
def _add(found: list[str], kind: str) -> None:
    if kind not in found:
        found.append(kind)


# Associa un chunk PNG al tipo di metadato che rappresenta (None se è un chunk ammesso)
def classify_chunk(ctype: bytes, payload: bytes) -> str | None:
    if ctype in ALLOWED_CHUNKS:
        return None
    if ctype == b"eXIf":
        return "exif"
    if ctype == b"iCCP":
        return "icc_profile"
    if ctype == b"caBX":
        return "c2pa"
    if ctype in (b"tEXt", b"zTXt", b"iTXt"):
        keyword = payload.split(b"\x00", 1)[0].lower()
        if keyword == b"xml:com.adobe.xmp":
            return "xmp"
        if b"iptc" in keyword:
            return "iptc"
        if b"exif" in keyword:
            return "exif"
        return "png_text"
    return "chunk_" + ctype.decode("latin-1")


# Scorre i chunk di un PNG e restituisce i metadati trovati (anche dati dopo IEND)
def scan_png_bytes(data: bytes) -> list[str]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("il file non è un PNG valido")
    found: list[str] = []
    pos = len(PNG_SIGNATURE)
    seen_end = False
    while pos + 12 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        start = pos + 8
        end = start + length
        if end + 4 > len(data):
            raise ValueError("chunk PNG troncato")
        payload = b"" if ctype == b"IDAT" else data[start:end]
        marker = classify_chunk(ctype, payload)
        if marker:
            _add(found, marker)
        pos = end + 4
        if ctype == b"IEND":
            seen_end = True
            break
    if not seen_end:
        raise ValueError("PNG senza chunk IEND")
    if pos < len(data):
        _add(found, "dati_dopo_IEND")
    return found


# Elenca i tipi di metadato presenti in un'immagine aperta e nei suoi byte grezzi
def detect_metadata(img: Image.Image, raw: bytes) -> list[str]:
    found: list[str] = []
    for key, value in img.info.items():
        if key in EXIF_KEYS:
            _add(found, "exif")
        elif key in XMP_KEYS:
            _add(found, "xmp")
        elif key in IPTC_KEYS:
            _add(found, "iptc")
        elif key == "icc_profile":
            _add(found, "icc_profile")
        elif key == "comment" and isinstance(value, bytes):
            _add(found, "jpeg_comment")
        elif isinstance(value, str):
            _add(found, "png_text")
    try:
        if len(img.getexif()) > 0:
            _add(found, "exif")
    except Exception as exc:
        logger.debug("lettura EXIF non riuscita: %s", exc)
    try:
        if IptcImagePlugin.getiptcinfo(img):
            _add(found, "iptc")
    except Exception as exc:
        logger.debug("lettura IPTC non riuscita: %s", exc)
    if raw.startswith(PNG_SIGNATURE):
        try:
            for marker in scan_png_bytes(raw):
                _add(found, marker)
        except ValueError as exc:
            logger.debug("scansione PNG non riuscita: %s", exc)
    elif b"jumb" in raw and b"c2pa" in raw:
        _add(found, "c2pa")
    return found


# Compone l'eventuale trasparenza su fondo bianco e converte in RGB
def flatten_to_rgb(img: Image.Image) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


# Ricrea l'immagine dai soli pixel, scartando ogni informazione associata
def rebuild_from_pixels(img: Image.Image) -> Image.Image:
    return Image.frombytes("RGB", img.size, img.tobytes())


# Legge un'immagine dai byte e restituisce l'RGB ripulito con l'elenco dei metadati rimossi
def clean_image_bytes(
    raw: bytes, max_pixels: int | None = None
) -> tuple[Image.Image, list[str]]:
    try:
        opened = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("il file non è un'immagine riconosciuta (JPEG, PNG o WebP)") from exc
    with opened:
        if opened.format not in ALLOWED_FORMATS:
            raise ValueError(f"formato non supportato: {opened.format} (usa JPEG, PNG o WebP)")
        if max_pixels is not None and opened.width * opened.height > max_pixels:
            raise ValueError(
                f"immagine troppo grande: {opened.width}x{opened.height} px "
                f"(massimo {max_pixels} pixel)"
            )
        opened.load()
        removed = detect_metadata(opened, raw)
        oriented = ImageOps.exif_transpose(opened)
        flat = flatten_to_rgb(oriented)
        cleaned = rebuild_from_pixels(flat)
    return cleaned, removed


# Apre un file e restituisce l'immagine RGB ripulita con l'elenco dei metadati rimossi
def clean_image(path: Path, max_pixels: int | None = None) -> tuple[Image.Image, list[str]]:
    return clean_image_bytes(path.read_bytes(), max_pixels)


# Controlla che i byte di un PNG non contengano metadati; altrimenti solleva MetadataError
def assert_clean_png_bytes(data: bytes, label: str = "PNG") -> None:
    try:
        leftovers = scan_png_bytes(data)
    except ValueError as exc:
        raise MetadataError(f"controllo metadati impossibile su {label}: {exc}") from exc
    if leftovers:
        raise MetadataError(f"metadati residui in {label}: {', '.join(leftovers)}")


# Controlla che un PNG su disco non contenga metadati
def assert_clean_png(path: Path) -> None:
    assert_clean_png_bytes(path.read_bytes(), path.name)


# Codifica l'immagine come PNG senza metadati e ne verifica il contenuto
def encode_clean_png(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    rebuild_from_pixels(img.convert("RGB")).save(buffer, format="PNG")
    data = buffer.getvalue()
    assert_clean_png_bytes(data)
    return data


# Scrive dei byte su disco senza mai sovrascrivere un file esistente
def save_png_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("xb")
    try:
        with handle:
            handle.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise


# Salva l'immagine come PNG senza metadati (mai sovrascrivendo) dopo averlo verificato
def save_clean_png(img: Image.Image, path: Path) -> None:
    save_png_bytes(encode_clean_png(img), path)
