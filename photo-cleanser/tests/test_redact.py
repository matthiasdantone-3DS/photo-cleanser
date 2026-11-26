import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings
from redactor.io_clean import scan_png_bytes
from redactor.pipeline import RedactionPipeline
from redactor.redact import expand_box, redact_image
from redactor.segmentation import SubjectMask

# Rilevamenti simulati: un volto dentro il soggetto, uno fuori e un logo sul soggetto
FACE_INSIDE = {"kind": "volto", "box": (55, 35, 70, 50), "score": 0.9}
FACE_OUTSIDE = {"kind": "volto", "box": (10, 10, 20, 20), "score": 0.8}
LOGO = {"kind": "logo", "box": (78, 56, 86, 64), "score": 0.7}


# Costruisce un'immagine sintetica (nessun pixel nero) con un soggetto rettangolare
def make_scene() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    image = rng.integers(1, 256, size=(80, 100, 3), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[30:70, 50:90] = True
    return image, mask


# Il box noto diventa nero, il soggetto resta identico e lo sfondo è nero
def test_box_black_and_subject_unchanged() -> None:
    image, mask = make_scene()
    detections = [{"kind": "logo", "box": (10, 10, 20, 20), "score": 0.9}]
    redacted, background_px = redact_image(image, detections, mask, pad_px=6)
    x1, y1, x2, y2 = expand_box((10, 10, 20, 20), 6, 100, 80)
    assert (x1, y1, x2, y2) == (4, 4, 26, 26)
    assert not redacted[y1:y2, x1:x2].any()
    assert np.array_equal(redacted[mask], image[mask])
    assert not redacted[~mask].any()
    assert background_px == int((~mask).sum())
    assert image[10:20, 10:20].all()


# Il margine non esce mai dai bordi dell'immagine
def test_padding_is_clipped_to_borders() -> None:
    assert expand_box((0, 0, 5, 5), 6, 100, 80) == (0, 0, 11, 11)
    assert expand_box((95, 75, 100, 80), 6, 100, 80) == (89, 69, 100, 80)


# Senza maschera non viene oscurato lo sfondo
def test_no_mask_means_no_background_redaction() -> None:
    image, _ = make_scene()
    redacted, background_px = redact_image(image, [], None, pad_px=6)
    assert background_px == 0
    assert np.array_equal(redacted, image)


# Crea una pipeline con segmentatore e rilevatori simulati, nessun modello scaricato
def build_pipeline(
    face_calls: list[list[dict]], mask_empty: bool = False
) -> tuple[RedactionPipeline, np.ndarray, np.ndarray, MagicMock]:
    image, mask = make_scene()
    segmenter = MagicMock()
    shown_mask = np.zeros_like(mask) if mask_empty else mask
    segmenter.segment.return_value = SubjectMask(mask=shown_mask, empty=mask_empty)
    faces = MagicMock()
    faces.detect.side_effect = face_calls
    logos = MagicMock()
    logos.detect.return_value = [LOGO]
    pipeline = RedactionPipeline(
        Settings(), segmenter=segmenter, face_detector=faces, logo_detector=logos
    )
    return pipeline, image, mask, faces


# Scrive l'immagine sorgente come PNG con EXIF
def write_source(tmp_path: Path, image: np.ndarray) -> Path:
    source = tmp_path / "foto.png"
    exif = Image.Exif()
    exif[0x010F] = "Camera"
    Image.fromarray(image).save(source, format="PNG", exif=exif)
    return source


# Caso completo: volto esterno e logo oscurati, volto del soggetto visibile, nessun metadato
def test_pipeline_end_to_end(tmp_path: Path) -> None:
    pipeline, image, mask, _ = build_pipeline([[FACE_INSIDE, FACE_OUTSIDE], [FACE_INSIDE]])
    source = write_source(tmp_path, image)
    result = pipeline.process(source, tmp_path / "out", set())
    assert result.status == "ok"
    assert result.detections == {"volto": 1, "logo": 1}
    assert "exif" in result.metadata_removed
    assert result.output is not None and result.output.endswith("foto_redacted.png")
    output = Path(result.output)
    assert scan_png_bytes(output.read_bytes()) == []
    expected = image.copy()
    expected[~mask] = 0
    expected[4:26, 4:26] = 0
    expected[50:70, 72:92] = 0
    assert np.array_equal(np.array(Image.open(output)), expected)


# Se la verifica finale trova ancora un volto fuori dal soggetto, lo stato è da_rivedere
def test_pipeline_flags_leftover_face(tmp_path: Path) -> None:
    pipeline, image, _, _ = build_pipeline([[FACE_INSIDE, FACE_OUTSIDE], [FACE_INSIDE, FACE_OUTSIDE]])
    result = pipeline.process(write_source(tmp_path, image), tmp_path / "out", set())
    assert result.status == "da_rivedere"
    assert result.output is not None and Path(result.output).is_file()


# Senza soggetto il file esce pulito dai metadati ma non oscurato
def test_pipeline_no_subject(tmp_path: Path) -> None:
    pipeline, image, _, faces = build_pipeline([], mask_empty=True)
    result = pipeline.process(write_source(tmp_path, image), tmp_path / "out", set())
    assert result.status == "nessun_soggetto"
    assert result.output is not None
    assert np.array_equal(np.array(Image.open(result.output)), image)
    faces.detect.assert_not_called()


# In dry-run non viene scritto nulla ma il report conta cosa verrebbe oscurato
def test_pipeline_dry_run_writes_nothing(tmp_path: Path) -> None:
    pipeline, image, mask, _ = build_pipeline([[FACE_INSIDE, FACE_OUTSIDE]])
    output_dir = tmp_path / "out"
    result = pipeline.process(write_source(tmp_path, image), output_dir, set(), dry_run=True)
    assert result.output is None
    assert not output_dir.exists()
    assert result.detections == {"volto": 1, "logo": 1}
    assert result.background_redacted_px == int((~mask).sum())


# Un file inesistente produce stato errore senza fermare la pipeline
def test_pipeline_error_is_reported(tmp_path: Path) -> None:
    pipeline, _, _, _ = build_pipeline([])
    result = pipeline.process(tmp_path / "manca.png", tmp_path / "out", set())
    assert result.status == "errore"
    assert result.errors


# L'elaborazione in memoria (usata dal sito) non scrive nulla su disco e rispetta il limite di pixel
def test_process_bytes_in_memory(tmp_path: Path) -> None:
    pipeline, image, mask, _ = build_pipeline([[FACE_INSIDE, FACE_OUTSIDE], [FACE_INSIDE]])
    source = write_source(tmp_path, image)
    png, result = pipeline.process_bytes(source.read_bytes(), "foto.png")
    assert result.status == "ok"
    assert png is not None and scan_png_bytes(png) == []
    assert result.output is None
    refused, failed = pipeline.process_bytes(source.read_bytes(), "foto.png", max_pixels=100)
    assert refused is None and failed.status == "errore"
