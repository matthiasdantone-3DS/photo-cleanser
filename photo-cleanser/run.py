from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from config import Settings
from redactor.pipeline import (
    STATUS_ERROR,
    RedactionPipeline,
    build_report,
    collect_inputs,
    run_batch,
    summarize,
    write_report,
)

logger = logging.getLogger("redactor")


# Definisce gli argomenti della riga di comando
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oscura volti, loghi e sfondo e rimuove i metadati dalle immagini"
    )
    parser.add_argument("--input", required=True, type=Path, help="file singolo o cartella (ricorsiva)")
    parser.add_argument("--output", required=True, type=Path, help="cartella di destinazione")
    parser.add_argument("--logo-model", type=Path, default=None, help="modello YOLO per i loghi (facoltativo)")
    parser.add_argument("--report", type=Path, default=None, help="scrive un report JSON")
    parser.add_argument("--dry-run", action="store_true", help="mostra cosa verrebbe oscurato senza scrivere immagini")
    parser.add_argument("--allow-unverified", action="store_true", help="accetta modelli senza checksum nel lock file")
    parser.add_argument("--face-score", type=float, default=None, help="soglia volti YuNet (default 0.6)")
    parser.add_argument("--logo-conf", type=float, default=None, help="soglia loghi YOLO (default 0.4)")
    parser.add_argument("--mask-threshold", type=float, default=None, help="soglia maschera soggetto (default 0.5)")
    parser.add_argument("--pad-px", type=int, default=None, help="margine in pixel attorno ai box (default 6)")
    return parser


# Applica ai default le soglie passate da riga di comando
def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {
        "face_score": args.face_score,
        "logo_conf": args.logo_conf,
        "mask_threshold": args.mask_threshold,
        "pad_px": args.pad_px,
    }
    return replace(Settings(), **{key: value for key, value in overrides.items() if value is not None})


# Punto di ingresso: restituisce 0 se nessun file è in errore, 1 altrimenti
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    for name in ("face_score", "logo_conf", "mask_threshold"):
        if not 0.0 <= getattr(settings, name) <= 1.0:
            parser.error(f"{name} deve essere compreso tra 0 e 1")
    if settings.pad_px < 0:
        parser.error("pad_px non può essere negativo")
    try:
        inputs = collect_inputs(args.input, args.output)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not inputs:
        logger.error("nessuna immagine trovata in %s", args.input)
        return 1
    pipeline = RedactionPipeline(
        settings, logo_model=args.logo_model, allow_unverified=args.allow_unverified
    )
    try:
        results = run_batch(pipeline, inputs, args.output, args.dry_run)
    finally:
        pipeline.close()
    if args.report:
        write_report(args.report, build_report(results, pipeline, args.dry_run))
        logger.info("report scritto in %s", args.report)
    counts = summarize(results)
    logger.info("riepilogo: %s", ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    return 1 if counts.get(STATUS_ERROR) else 0


if __name__ == "__main__":
    sys.exit(main())
