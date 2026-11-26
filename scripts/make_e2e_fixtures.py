from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

STRIDES = (8, 16, 32)
SIZE = 512
# Volti del modello sintetico: (centro x, centro y, lato) in pixel di un'immagine 512x512
FACES = [(232, 96, 64), (450, 60, 64)]


# Inserisce un segmento APP1 con un pacchetto XMP subito dopo l'inizio di un JPEG
def add_xmp(jpeg: bytes) -> bytes:
    packet = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF/></x:xmpmeta>"
    segment = b"\xff\xe1" + (len(packet) + 2).to_bytes(2, "big") + packet
    return jpeg[:2] + segment + jpeg[2:]


# Crea le immagini di prova con metadati noti
def make_images(source: Path, out: Path) -> None:
    base = Image.open(source).convert("RGB")
    base.save(out / "astronaut.png")
    exif = Image.Exif()
    exif[0x010F] = "FotoCamera"
    exif[0x0110] = "Modello-X"
    exif[0x0112] = 1
    buffer = io.BytesIO()
    base.save(buffer, format="JPEG", quality=92, exif=exif, comment=b"commento privato")
    (out / "astronaut_exif.jpg").write_bytes(add_xmp(buffer.getvalue()))
    rotated_exif = Image.Exif()
    rotated_exif[0x0112] = 6
    Image.new("RGB", (40, 20), (200, 30, 30)).save(out / "orientamento_6.jpg", format="JPEG", exif=rotated_exif)
    (out / "non_immagine.png").write_bytes(b"questo non e' un'immagine")


# Crea un YuNet sintetico a 12 uscite con volti in posizioni note, a ingresso fisso 512x512
def make_face_model(out: Path) -> None:
    outputs: dict[str, np.ndarray] = {}
    for stride in STRIDES:
        cols = SIZE // stride
        cells = cols * cols
        outputs[f"cls_{stride}"] = np.zeros((1, cells, 1), dtype=np.float32)
        outputs[f"obj_{stride}"] = np.zeros((1, cells, 1), dtype=np.float32)
        bbox = np.zeros((1, cells, 4), dtype=np.float32)
        outputs[f"kps_{stride}"] = np.zeros((1, cells, 10), dtype=np.float32)
        if stride == 16:
            for cx, cy, side in FACES:
                column, row = int(cx // stride - 1 + 1), int(cy // stride)
                column = int(cx / stride - 0.5)
                row = int(cy / stride - 0.5)
                index = row * cols + column
                outputs["cls_16"][0, index, 0] = 0.95
                outputs["obj_16"][0, index, 0] = 0.95
                bbox[0, index, 0] = cx / stride - column
                bbox[0, index, 1] = cy / stride - row
                bbox[0, index, 2:4] = math.log(side / stride)
        outputs[f"bbox_{stride}"] = bbox
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zero_signal"]),
    ]
    initializers = [numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")]
    graph_outputs = []
    for name in [f"{kind}_{s}" for kind in ("cls", "obj", "bbox", "kps") for s in STRIDES]:
        initializers.append(numpy_helper.from_array(outputs[name], name=f"{name}_const"))
        nodes.append(helper.make_node("Add", [f"{name}_const", "zero_signal"], [name]))
        graph_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(outputs[name].shape)))
    graph = helper.make_graph(
        nodes,
        "yunet_sintetico_e2e",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out / "yunet_sintetico_e2e.onnx")


# Genera le immagini e il modello sintetico usati dai test nel browser
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixture per i test end-to-end")
    parser.add_argument("--source", type=Path, required=True, help="immagine di partenza 512x512 (es. skimage.data.astronaut)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests" / "fixtures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    make_images(args.source, args.out)
    make_face_model(args.out)
    print("fixture scritte in", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
