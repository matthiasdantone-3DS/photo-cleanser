from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Percorsi principali del progetto
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
LOCK_FILE = MODELS_DIR / "models.lock.json"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_LOGO_PATH = MODELS_DIR / "logo.pt"

# Chiavi dei modelli nel lock file
YUNET_KEY = "yunet"
LOGO_KEY = "logo"

# Formati immagine accettati quando si scorre una cartella
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


# Tutte le soglie del programma, con i valori di default richiesti
@dataclass(frozen=True)
class Settings:
    face_score: float = 0.6
    logo_conf: float = 0.4
    mask_threshold: float = 0.5
    mask_min_component_ratio: float = 0.02
    pad_px: int = 6
    morph_kernel: int = 5


# Errore sollevato quando un modello non supera il controllo di integrità
class ModelIntegrityError(RuntimeError):
    pass


# Calcola lo SHA-256 di un file leggendolo a blocchi
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Legge il lock file dei modelli
def load_lock(lock_path: Path = LOCK_FILE) -> dict[str, Any]:
    with lock_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Salva il lock file con formattazione stabile
def _save_lock(lock: dict[str, Any], lock_path: Path) -> None:
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Scrive nel lock lo SHA-256 di un modello, con una nota
def write_lock_checksum(
    key: str, checksum: str, note: str, lock_path: Path = LOCK_FILE
) -> None:
    lock = load_lock(lock_path)
    entry = lock.setdefault("models", {}).setdefault(key, {})
    entry["sha256"] = checksum
    entry["sha256_note"] = note
    _save_lock(lock, lock_path)


# Scrive nel lock l'URL di download di un modello
def write_lock_url(key: str, url: str, lock_path: Path = LOCK_FILE) -> None:
    lock = load_lock(lock_path)
    lock.setdefault("models", {}).setdefault(key, {})["url"] = url
    _save_lock(lock, lock_path)


# Restituisce la voce del lock per un modello (vuota se manca o il file è illeggibile)
def lock_entry(key: str, lock_path: Path = LOCK_FILE) -> dict[str, Any]:
    try:
        models = load_lock(lock_path).get("models", {})
    except (OSError, ValueError) as exc:
        logger.warning("lock file non leggibile (%s): nessun checksum disponibile", exc)
        return {}
    return dict(models.get(key, {}))


# Rifiuta modelli senza checksum o con checksum diverso, salvo allow_unverified
def verify_model(
    path: Path,
    key: str,
    allow_unverified: bool = False,
    lock_path: Path = LOCK_FILE,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"modello non trovato: {path}")
    expected = str(lock_entry(key, lock_path).get("sha256", "")).strip().lower()
    actual = sha256_file(path)
    if not expected:
        if allow_unverified:
            logger.warning(
                "modello %s usato senza checksum verificato (sha256 calcolato: %s)",
                path.name,
                actual,
            )
            return
        raise ModelIntegrityError(
            f"checksum mancante in models.lock.json per '{key}' "
            f"(sha256 calcolato: {actual}); verificalo e inseriscilo nel lock, "
            f"oppure usa --allow-unverified"
        )
    if actual != expected:
        raise ModelIntegrityError(
            f"checksum non corrispondente per '{key}': atteso {expected}, trovato {actual}"
        )
