from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Literal

# Rende importabile config.py anche lanciando lo script da models/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MODELS_DIR, load_lock, sha256_file, write_lock_checksum, write_lock_url

logger = logging.getLogger("download_models")

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


# Scarica l'URL (solo HTTPS) nel file indicato e restituisce lo SHA-256 calcolato al volo
def download_to(url: str, target: Path) -> str:
    if not url.lower().startswith("https://"):
        raise ValueError(f"URL non HTTPS rifiutato: {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(block)
            digest.update(block)
    return digest.hexdigest()


# Scarica un modello e controlla lo SHA-256; se non coincide elimina il file e fallisce
def fetch_model(key: str, entry: dict[str, Any], allow_unverified: bool) -> None:
    url = str(entry.get("url", "")).strip()
    if not url:
        logger.info("modello '%s' fornito dall'utente: nessun download", key)
        return
    expected = str(entry.get("sha256", "")).strip().lower()
    if not expected and not allow_unverified:
        raise RuntimeError(
            f"checksum mancante per '{key}': download rifiutato "
            f"(alla prima esecuzione usa --allow-unverified)"
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / str(entry["file"])
    if target.is_file():
        if expected and sha256_file(target) == expected:
            logger.info("già presente e verificato: %s", target.name)
            return
        target.unlink()
    temp = target.with_name(target.name + ".part")
    try:
        actual = download_to(url, temp)
        if expected and actual != expected:
            raise RuntimeError(
                f"SHA-256 non corrispondente per '{key}': atteso {expected}, ottenuto {actual}"
            )
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if not expected:
        write_lock_checksum(
            key,
            actual,
            "compilato alla prima esecuzione: verifica che corrisponda alla fonte ufficiale",
        )
        logger.warning(
            "'%s' scaricato senza checksum di riferimento: sha256 %s salvato nel lock, "
            "verificalo sulla fonte ufficiale",
            key,
            actual,
        )
    logger.info("scaricato e verificato: %s", target.name)


# Interpreta le coppie CHIAVE=VALORE di --pin (sha256) e --url (solo https) controllando i valori
def parse_pairs(
    values: list[str], known: set[str], kind: Literal["sha256", "url"]
) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        key, separator, content = value.partition("=")
        if not separator or not content:
            raise ValueError(f"formato non valido '{value}': usa CHIAVE=VALORE")
        if key not in known:
            raise ValueError(f"modello sconosciuto '{key}' (disponibili: {', '.join(sorted(known))})")
        if kind == "sha256":
            if not SHA256_PATTERN.match(content):
                raise ValueError(f"sha256 non valido per '{key}': servono 64 caratteri esadecimali")
            content = content.lower()
        elif not content.lower().startswith("https://"):
            raise ValueError(f"URL per '{key}' rifiutato: deve iniziare con https://")
        pairs[key] = content
    return pairs


# Punto di ingresso: scarica tutti i modelli del lock e restituisce il codice di uscita
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Scarica e verifica i modelli pre-addestrati")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="scarica anche i modelli senza checksum e salva nel lock quello calcolato",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="CHIAVE=SHA256",
        help="fissa nel lock lo SHA-256 atteso di un modello (ripetibile)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="CHIAVE=URL",
        help="imposta nel lock l'URL https di un modello (ripetibile)",
    )
    args = parser.parse_args(argv)
    known = set(load_lock().get("models", {}))
    try:
        urls = parse_pairs(args.url, known, "url")
        pins = parse_pairs(args.pin, known, "sha256")
    except ValueError as exc:
        parser.error(str(exc))
    for key, url in urls.items():
        write_lock_url(key, url)
    for key, checksum in pins.items():
        write_lock_checksum(key, checksum, "fissato con --pin")
    lock = load_lock()
    failures = 0
    for key, entry in lock.get("models", {}).items():
        try:
            fetch_model(key, entry, args.allow_unverified)
        except Exception as exc:
            logger.error("%s: %s", key, exc)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
