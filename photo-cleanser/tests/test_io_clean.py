import sys
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redactor.io_clean import (
    MetadataError,
    assert_clean_png,
    clean_image,
    save_clean_png,
    scan_png_bytes,
)


# Crea un PNG rosso con EXIF, testo PNG e XMP
def make_dirty_png(path: Path) -> None:
    image = Image.new("RGB", (64, 48), (200, 30, 30))
    exif = Image.Exif()
    exif[0x010F] = "FotoCamera"
    exif[0x0110] = "Modello-X"
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Mario Rossi")
    info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta xmlns:x='adobe:ns:meta/'></x:xmpmeta>")
    image.save(path, format="PNG", exif=exif, pnginfo=info)


# Il PNG di partenza ha davvero i metadati, quindi il test è significativo
def test_source_has_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sporco.png"
    make_dirty_png(source)
    found = scan_png_bytes(source.read_bytes())
    assert "exif" in found
    assert "png_text" in found
    assert "xmp" in found
    with pytest.raises(MetadataError):
        assert_clean_png(source)


# Dopo la pulizia il file salvato non contiene EXIF, XMP né testo PNG
def test_clean_removes_all_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sporco.png"
    make_dirty_png(source)
    cleaned, removed = clean_image(source)
    assert {"exif", "png_text", "xmp"} <= set(removed)
    output = tmp_path / "pulito.png"
    save_clean_png(cleaned, output)
    assert scan_png_bytes(output.read_bytes()) == []
    assert_clean_png(output)
    with Image.open(output) as reopened:
        assert len(reopened.getexif()) == 0
        assert not any(isinstance(value, str) for value in reopened.info.values())
        assert reopened.getpixel((0, 0)) == (200, 30, 30)


# La trasparenza viene composta su fondo bianco
def test_transparency_goes_on_white(tmp_path: Path) -> None:
    source = tmp_path / "alpha.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(source, format="PNG")
    cleaned, _ = clean_image(source)
    assert cleaned.mode == "RGB"
    assert cleaned.getpixel((0, 0)) == (255, 255, 255)


# L'orientamento EXIF viene applicato prima di eliminare i metadati
def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    source = tmp_path / "ruotata.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), (10, 10, 10)).save(source, format="JPEG", exif=exif)
    cleaned, removed = clean_image(source)
    assert cleaned.size == (20, 40)
    assert "exif" in removed


# Il salvataggio non sovrascrive file esistenti
def test_save_does_not_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "esistente.png"
    target.write_bytes(b"originale")
    with pytest.raises(FileExistsError):
        save_clean_png(Image.new("RGB", (2, 2)), target)
    assert target.read_bytes() == b"originale"


# File non immagine, formati non ammessi e immagini troppo grandi vengono rifiutati
def test_rejects_invalid_inputs(tmp_path: Path) -> None:
    bogus = tmp_path / "finto.png"
    bogus.write_bytes(b"non sono un'immagine")
    with pytest.raises(ValueError):
        clean_image(bogus)
    gif = tmp_path / "animata.gif"
    Image.new("RGB", (4, 4)).save(gif, format="GIF")
    with pytest.raises(ValueError):
        clean_image(gif)
    big = tmp_path / "grande.png"
    Image.new("RGB", (100, 100)).save(big, format="PNG")
    with pytest.raises(ValueError):
        clean_image(big, max_pixels=1000)
