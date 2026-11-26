from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

STRIDES = (8, 16, 32)


# Costruisce le uscite casuali di un finto YuNet: celle di griglia per stride 8, 16 e 32
def make_outputs(width: int, height: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    outputs: dict[str, np.ndarray] = {}
    for stride in STRIDES:
        cells = (height // stride) * (width // stride)
        cls = rng.uniform(0.0, 0.5, size=(1, cells, 1)).astype(np.float32)
        obj = rng.uniform(0.0, 0.5, size=(1, cells, 1)).astype(np.float32)
        bbox = np.zeros((1, cells, 4), dtype=np.float32)
        bbox[..., 0:2] = rng.uniform(-0.5, 0.5, size=(1, cells, 2))
        bbox[..., 2:4] = rng.uniform(0.4, 1.6, size=(1, cells, 2))
        kps = rng.uniform(-1.5, 1.5, size=(1, cells, 10)).astype(np.float32)
        hot = rng.choice(cells, size=max(4, cells // 40), replace=False)
        cls[0, hot, 0] = rng.uniform(0.7, 1.0, size=hot.size)
        obj[0, hot, 0] = rng.uniform(0.7, 1.0, size=hot.size)
        cls[0, rng.choice(cells, size=3), 0] = 1.7
        obj[0, rng.choice(cells, size=3), 0] = -0.4
        outputs[f"cls_{stride}"] = cls
        outputs[f"obj_{stride}"] = obj
        outputs[f"bbox_{stride}"] = bbox
        outputs[f"kps_{stride}"] = kps
    return outputs


# Crea un ONNX con ingresso BGR [1,3,H,W] e le 12 uscite di YuNet 2023mar, costanti ma legate all'ingresso
def build_model(outputs: dict[str, np.ndarray], width: int, height: int) -> onnx.ModelProto:
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zero_signal"]),
    ]
    initializers = [numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")]
    graph_outputs = []
    order = [f"{kind}_{s}" for kind in ("cls", "obj", "bbox", "kps") for s in STRIDES]
    for name in order:
        initializers.append(numpy_helper.from_array(outputs[name], name=f"{name}_const"))
        nodes.append(helper.make_node("Add", [f"{name}_const", "zero_signal"], [name]))
        graph_outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(outputs[name].shape))
        )
    graph = helper.make_graph(
        nodes,
        "yunet_sintetico",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])],
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# Genera il modello, lo fa leggere a OpenCV (FaceDetectorYN) e salva i risultati di riferimento
def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description="Modello YuNet sintetico per il test della decodifica")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cases = []
    for seed in range(args.seeds):
        outputs = make_outputs(args.width, args.height, seed)
        model_path = args.out / f"yunet_sintetico_{seed}.onnx"
        onnx.save(build_model(outputs, args.width, args.height), model_path)
        for score_thr, nms_thr in ((0.6, 0.3), (0.9, 0.3), (0.3, 0.5)):
            detector = cv2.FaceDetectorYN.create(
                str(model_path), "", (args.width, args.height), score_thr, nms_thr, 5000
            )
            image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            _, faces = detector.detect(image)
            cases.append(
                {
                    "model": model_path.name,
                    "score_threshold": score_thr,
                    "nms_threshold": nms_thr,
                    "faces": [] if faces is None else [[float(v) for v in row] for row in faces],
                }
            )
    (args.out / "reference.json").write_text(
        json.dumps({"width": args.width, "height": args.height, "cases": cases}), encoding="utf-8"
    )
    print("casi:", len(cases), "- volti per caso:", [len(c["faces"]) for c in cases])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
