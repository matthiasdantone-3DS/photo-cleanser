import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.prepare_dataset import main as prepare_main
from training.voc_to_yolo import main as voc_main


# Crea una cartella sorgente con immagini piccole ed etichette YOLO
def make_source(root: Path, count: int = 10, bad_class: bool = False) -> Path:
    source = root / "source"
    source.mkdir()
    for number in range(count):
        Image.new("RGB", (16, 16), (number * 10, 50, 50)).save(source / f"img{number}.jpg")
        if number == count - 1:
            continue
        class_id = 5 if (bad_class and number == 0) else 0
        (source / f"img{number}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return source


# Il dataset viene diviso in train e val, con data.yaml e un esempio negativo
def test_prepare_dataset(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    dest = tmp_path / "dataset"
    assert prepare_main(["--source", str(source), "--dest", str(dest), "--seed", "1"]) == 0
    train_images = list((dest / "images" / "train").iterdir())
    val_images = list((dest / "images" / "val").iterdir())
    assert len(train_images) + len(val_images) == 10
    assert len(val_images) == 2
    assert len(list((dest / "labels" / "train").iterdir())) == len(train_images)
    yaml_text = (dest / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert '0: "logo"' in yaml_text


# Un'etichetta con classe inesistente blocca la preparazione senza copiare nulla
def test_prepare_dataset_rejects_bad_labels(tmp_path: Path) -> None:
    source = make_source(tmp_path, bad_class=True)
    dest = tmp_path / "dataset"
    assert prepare_main(["--source", str(source), "--dest", str(dest)]) == 1
    assert not dest.exists()


# Un dataset già presente non viene sovrascritto senza --force
def test_prepare_dataset_needs_force(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    dest = tmp_path / "dataset"
    assert prepare_main(["--source", str(source), "--dest", str(dest)]) == 0
    assert prepare_main(["--source", str(source), "--dest", str(dest)]) == 1
    assert prepare_main(["--source", str(source), "--dest", str(dest), "--force"]) == 0


# La conversione VOC produce coordinate YOLO normalizzate
def test_voc_to_yolo(tmp_path: Path) -> None:
    source = tmp_path / "voc"
    source.mkdir()
    Image.new("RGB", (100, 50), (1, 2, 3)).save(source / "foto.jpg")
    (source / "foto.xml").write_text(
        "<annotation><size><width>100</width><height>50</height></size>"
        "<object><name>acme</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
        "<xmax>50</xmax><ymax>30</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    dest = tmp_path / "yolo"
    assert voc_main(["--source", str(source), "--dest", str(dest)]) == 0
    label = next(dest.glob("0*.txt"))
    values = label.read_text(encoding="utf-8").split()
    assert values[0] == "0"
    assert [round(float(value), 3) for value in values[1:]] == [0.3, 0.4, 0.4, 0.4]
    assert (dest / "classes.txt").read_text(encoding="utf-8").strip() == "logo"
