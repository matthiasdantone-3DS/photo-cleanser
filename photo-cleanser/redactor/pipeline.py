from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import (
    LOGO_KEY,
    SUPPORTED_EXTENSIONS,
    YUNET_KEY,
    YUNET_PATH,
    Settings,
    verify_model,
)
from redactor.detectors import (
    KIND_FACE,
    KIND_LOGO,
    Detection,
    Detector,
    FaceDetector,
    LogoDetector,
)
from redactor.io_clean import (
    MetadataError,
    assert_clean_png_bytes,
    clean_image_bytes,
    encode_clean_png,
    save_png_bytes,
)
from redactor.redact import redact_image
from redactor.segmentation import Segmenter, SubjectSegmenter, center_inside_mask

logger = logging.getLogger(__name__)

# Stati possibili di un file nel report
STATUS_OK = "ok"
STATUS_NO_SUBJECT = "nessun_soggetto"
STATUS_REVIEW = "da_rivedere"
STATUS_ERROR = "errore"


# Risultato dell'elaborazione di un singolo file, serializzabile nel report
@dataclass
class FileResult:
    input: str
    output: str | None = None
    status: str = STATUS_OK
    detections: dict[str, int] = field(default_factory=lambda: {KIND_FACE: 0, KIND_LOGO: 0})
    detection_details: list[dict[str, Any]] = field(default_factory=list)
    background_redacted_px: int = 0
    metadata_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    # Converte il risultato nel formato del report JSON
    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "detections": self.detections,
            "background_redacted_px": self.background_redacted_px,
            "metadata_removed": self.metadata_removed,
            "errors": self.errors,
            "detection_details": self.detection_details,
            "elapsed_s": self.elapsed_s,
        }


# Sceglie un nome libero in output aggiungendo un suffisso numerico se serve
def unique_output_path(output_dir: Path, source: Path, reserved: set[Path]) -> Path:
    candidate = output_dir / f"{source.stem}_redacted.png"
    counter = 1
    while candidate.exists() or candidate in reserved:
        candidate = output_dir / f"{source.stem}_redacted_{counter}.png"
        counter += 1
    reserved.add(candidate)
    return candidate


# Elenca le immagini da elaborare (file singolo o cartella ricorsiva), escludendo l'output
def collect_inputs(source: Path, exclude_dir: Path | None = None) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"input non trovato: {source}")
    excluded = exclude_dir.resolve() if exclude_dir else None
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if excluded is not None and excluded in path.resolve().parents:
            continue
        files.append(path)
    return files


# Pipeline completa: pulizia, segmentazione, rilevamento, oscuramento, verifica
class RedactionPipeline:
    def __init__(
        self,
        settings: Settings,
        logo_model: Path | None = None,
        allow_unverified: bool = False,
        segmenter: Segmenter | None = None,
        face_detector: Detector | None = None,
        logo_detector: Detector | None = None,
    ) -> None:
        self.settings = settings
        self.logo_model = logo_model
        self.skipped_phases: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._phase_errors: dict[str, str] = {}
        self._segmenter_error: str | None = None
        self._segmenter = segmenter
        if self._segmenter is None:
            try:
                self._segmenter = SubjectSegmenter(settings)
            except Exception as exc:
                self._segmenter_error = f"segmentazione non disponibile: {exc}"
                logger.error(self._segmenter_error)
        self._face_detector = face_detector
        if face_detector is None:
            self._face_detector = self._load_faces(allow_unverified)
        self._logo_detector = logo_detector
        if logo_detector is None:
            self._logo_detector = self._load_logos(allow_unverified)

    # Registra una fase saltata per errore, nel report e nel log
    def _register_failure(self, phase: str, message: str) -> None:
        self._phase_errors[phase] = message
        self.skipped_phases.append({"phase": phase, "reason": message})
        logger.error("fase %s saltata: %s", phase, message)

    # Carica YuNet dopo il controllo del checksum
    def _load_faces(self, allow_unverified: bool) -> Detector | None:
        try:
            verify_model(YUNET_PATH, YUNET_KEY, allow_unverified)
            return FaceDetector(YUNET_PATH, self.settings.face_score)
        except Exception as exc:
            self._register_failure(KIND_FACE, str(exc))
            return None

    # Carica il modello loghi se fornito; altrimenti salta la fase senza errori
    def _load_logos(self, allow_unverified: bool) -> Detector | None:
        if self.logo_model is None:
            self.skipped_phases.append(
                {"phase": KIND_LOGO, "reason": "nessun modello fornito (--logo-model assente)"}
            )
            logger.info("fase loghi saltata: nessun modello fornito")
            return None
        try:
            verify_model(self.logo_model, LOGO_KEY, allow_unverified)
            return LogoDetector(self.logo_model, self.settings.logo_conf)
        except Exception as exc:
            self._register_failure(KIND_LOGO, str(exc))
            return None

    # Dice quali fasi sono attive, per il sito e per il report
    def capabilities(self) -> dict[str, Any]:
        return {
            "segmentation": self._segmenter is not None,
            "faces": self._face_detector is not None,
            "logos": self._logo_detector is not None,
            "skipped_phases": self.skipped_phases,
            "segmenter_error": self._segmenter_error,
        }

    # Libera le risorse del segmentatore
    def close(self) -> None:
        closer = getattr(self._segmenter, "close", None)
        if callable(closer):
            closer()

    # Elabora un file su disco e restituisce sempre un risultato, anche in caso di errore
    def process(
        self,
        source: Path,
        output_dir: Path,
        reserved: set[Path],
        dry_run: bool = False,
    ) -> FileResult:
        result = FileResult(input=str(source))
        started = time.perf_counter()
        try:
            png = self._run(source.read_bytes(), source.name, dry_run, result)
            target = unique_output_path(output_dir, source, reserved)
            if png is None:
                logger.info("[dry-run] output previsto: %s", target)
            else:
                save_png_bytes(png, target)
                result.output = str(target)
        except Exception as exc:
            self._fail(result, exc, source.name)
        result.elapsed_s = round(time.perf_counter() - started, 3)
        return result

    # Elabora un'immagine in memoria (usato dal sito): niente viene scritto su disco
    def process_bytes(
        self, raw: bytes, name: str, max_pixels: int | None = None
    ) -> tuple[bytes | None, FileResult]:
        result = FileResult(input=name)
        started = time.perf_counter()
        png: bytes | None = None
        try:
            png = self._run(raw, "upload", False, result, max_pixels)
        except Exception as exc:
            self._fail(result, exc, "upload")
        result.elapsed_s = round(time.perf_counter() - started, 3)
        return png, result

    # Segna un file come in errore con un messaggio leggibile
    def _fail(self, result: FileResult, exc: Exception, label: str) -> None:
        result.status = STATUS_ERROR
        message = str(exc) if isinstance(exc, (ValueError, MetadataError)) else (
            f"{type(exc).__name__}: {exc}"
        )
        result.errors.append(message)
        logger.error("%s: errore: %s", label, message)

    # Passi della pipeline su byte grezzi: restituisce il PNG finale (None in dry-run)
    def _run(
        self,
        raw: bytes,
        label: str,
        dry_run: bool,
        result: FileResult,
        max_pixels: int | None = None,
    ) -> bytes | None:
        with self._lock:
            cleaned, removed = clean_image_bytes(raw, max_pixels)
            result.metadata_removed = removed
            image, mask = self._redact(cleaned, result, label, dry_run)
            if image is None:
                return None
            png = encode_clean_png(image)
            self._verify_png(png, mask, result)
            return png

    # Segmenta, rileva e oscura: restituisce l'immagine finale e la maschera del soggetto
    def _redact(
        self, cleaned: Image.Image, result: FileResult, label: str, dry_run: bool
    ) -> tuple[Image.Image | None, np.ndarray | None]:
        image_rgb = np.array(cleaned)
        if self._segmenter is None:
            raise RuntimeError(self._segmenter_error or "segmentatore non disponibile")
        subject = self._segmenter.segment(image_rgb)
        if subject.empty:
            result.status = STATUS_NO_SUBJECT
            logger.warning("%s: nessun soggetto rilevato, nulla viene oscurato", label)
            return (None if dry_run else cleaned), None
        redactions = self._detect_all(image_rgb, subject.mask, result)
        result.background_redacted_px = int((~subject.mask).sum())
        self._log_summary(label, result, dry_run)
        if dry_run:
            return None, subject.mask
        redacted, _ = redact_image(image_rgb, redactions, subject.mask, self.settings.pad_px)
        return Image.fromarray(redacted), subject.mask

    # Esegue i rilevatori e restituisce solo i riquadri da oscurare
    def _detect_all(
        self, image_rgb: np.ndarray, mask: np.ndarray, result: FileResult
    ) -> list[Detection]:
        redactions: list[Detection] = []
        for face in self._run_detector(KIND_FACE, self._face_detector, image_rgb, result):
            inside = center_inside_mask(face["box"], mask)
            self._record(result, face, redacted=not inside)
            if not inside:
                redactions.append(face)
        for logo in self._run_detector(KIND_LOGO, self._logo_detector, image_rgb, result):
            self._record(result, logo, redacted=True)
            redactions.append(logo)
        return redactions

    # Lancia un rilevatore; un suo errore viene registrato e la fase saltata
    def _run_detector(
        self,
        kind: str,
        detector: Detector | None,
        image_rgb: np.ndarray,
        result: FileResult,
    ) -> list[Detection]:
        if detector is None:
            reason = self._phase_errors.get(kind)
            if reason:
                self._flag_review(result, f"{kind}: fase saltata ({reason})")
            return []
        try:
            return detector.detect(image_rgb)
        except Exception as exc:
            logger.error("rilevatore %s fallito: %s", kind, exc)
            self._flag_review(result, f"{kind}: errore del rilevatore ({exc})")
            return []

    # Annota un rilevamento nel risultato e aggiorna i conteggi di quelli oscurati
    def _record(self, result: FileResult, detection: Detection, redacted: bool) -> None:
        result.detection_details.append(
            {
                "kind": detection["kind"],
                "box": list(detection["box"]),
                "score": round(detection["score"], 4),
                "redacted": redacted,
            }
        )
        if redacted:
            result.detections[detection["kind"]] += 1

    # Segna il file come da rivedere e registra il motivo
    def _flag_review(self, result: FileResult, message: str) -> None:
        result.errors.append(message)
        if result.status == STATUS_OK:
            result.status = STATUS_REVIEW

    # Scrive il riepilogo del file nel log (e i dettagli in modalità dry-run)
    def _log_summary(self, label: str, result: FileResult, dry_run: bool) -> None:
        prefix = "[dry-run] verrebbero oscurati" if dry_run else "oscurati"
        logger.info(
            "%s: %s %d volti, %d loghi, sfondo %d px",
            label,
            prefix,
            result.detections[KIND_FACE],
            result.detections[KIND_LOGO],
            result.background_redacted_px,
        )
        if dry_run:
            for detail in result.detection_details:
                logger.info(
                    "  %s box=%s score=%.2f %s",
                    detail["kind"],
                    tuple(detail["box"]),
                    detail["score"],
                    "oscurato" if detail["redacted"] else "visibile (soggetto)",
                )

    # Rilegge il PNG finale: nessun metadato e nessun volto fuori dal soggetto
    def _verify_png(self, png: bytes, mask: np.ndarray | None, result: FileResult) -> None:
        assert_clean_png_bytes(png)
        if mask is None or self._face_detector is None:
            return
        with Image.open(io.BytesIO(png)) as saved:
            saved_rgb = np.array(saved.convert("RGB"))
        try:
            found = self._face_detector.detect(saved_rgb)
        except Exception as exc:
            self._flag_review(result, f"verifica finale dei volti non riuscita ({exc})")
            return
        leftovers = [face for face in found if not center_inside_mask(face["box"], mask)]
        if leftovers:
            self._flag_review(
                result, f"{len(leftovers)} volti non coperti ancora rilevati nell'output"
            )


# Elabora una lista di file e restituisce i risultati
def run_batch(
    pipeline: RedactionPipeline,
    inputs: list[Path],
    output_dir: Path,
    dry_run: bool = False,
) -> list[FileResult]:
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    reserved: set[Path] = set()
    results: list[FileResult] = []
    for index, source in enumerate(inputs, start=1):
        logger.info("[%d/%d] %s", index, len(inputs), source)
        results.append(pipeline.process(source, output_dir, reserved, dry_run))
    return results


# Costruisce il dizionario del report JSON
def build_report(
    results: list[FileResult], pipeline: RedactionPipeline, dry_run: bool
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "settings": asdict(pipeline.settings),
        "logo_model": str(pipeline.logo_model) if pipeline.logo_model else None,
        "skipped_phases": pipeline.skipped_phases,
        "files": [item.to_dict() for item in results],
    }


# Scrive il report su file in formato JSON
def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


# Conta i file per stato, utile per il riepilogo finale
def summarize(results: list[FileResult]) -> Counter[str]:
    return Counter(item.status for item in results)
