import { nms } from "./nms.js";
import { resizeRgba } from "./imageops.js";

const PAD_VALUE = 114;

// Decodifica l'uscita di un modello YOLO esportato in ONNX: classico [1, 4+classi, N] oppure end-to-end [1, N, 6]
export function decodeYolo(data, dims, { conf, iou, scale, padLeft, padTop, width, height }) {
  const boxes = [];
  const scores = [];
  const classes = [];
  if (dims.length === 3 && dims[2] === 6) {
    for (let i = 0; i < dims[1]; i += 1) {
      const row = i * 6;
      if (data[row + 4] > conf) {
        boxes.push([data[row], data[row + 1], data[row + 2], data[row + 3]]);
        scores.push(data[row + 4]);
        classes.push(data[row + 5]);
      }
    }
  } else {
    const channels = dims[1];
    const count = dims[2];
    for (let i = 0; i < count; i += 1) {
      let best = -1;
      let bestScore = 0;
      for (let c = 4; c < channels; c += 1) {
        const value = data[c * count + i];
        if (value > bestScore) {
          bestScore = value;
          best = c - 4;
        }
      }
      if (bestScore > conf) {
        const cx = data[i];
        const cy = data[count + i];
        const w = data[2 * count + i];
        const h = data[3 * count + i];
        boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
        scores.push(bestScore);
        classes.push(best);
      }
    }
  }
  const kept = [];
  for (const cls of new Set(classes)) {
    const indexes = classes.map((c, i) => (c === cls ? i : -1)).filter((i) => i >= 0);
    const local = nms(indexes.map((i) => boxes[i]), indexes.map((i) => scores[i]), { scoreThreshold: conf, iouThreshold: iou });
    for (const k of local) {
      kept.push(indexes[k]);
    }
  }
  kept.sort((a, b) => scores[b] - scores[a]);
  return kept.map((i) => {
    const [x1, y1, x2, y2] = boxes[i];
    const box = [
      Math.max(0, Math.floor((x1 - padLeft) / scale)),
      Math.max(0, Math.floor((y1 - padTop) / scale)),
      Math.min(width, Math.ceil((x2 - padLeft) / scale)),
      Math.min(height, Math.ceil((y2 - padTop) / scale)),
    ];
    return { kind: "logo", box, score: scores[i] };
  }).filter((d) => d.box[2] > d.box[0] && d.box[3] > d.box[1]);
}

// Rilevatore di loghi: modello YOLO (addestrato con training/) esportato in ONNX
export class LogoDetector {
  constructor(ort, session, options) {
    this.ort = ort;
    this.session = session;
    this.conf = options.conf ?? 0.4;
    this.iou = options.iou ?? 0.7;
    const meta = session.inputMetadata ? session.inputMetadata[0] : null;
    const shape = meta && meta.shape ? meta.shape : [];
    this.size = Number.isInteger(shape[2]) && shape[2] > 0 ? shape[2] : 640;
  }

  // Crea il rilevatore dal contenuto del file ONNX
  static async create(ort, modelBytes, options = {}) {
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    return new LogoDetector(ort, session, options);
  }

  // Trova i loghi in un'immagine RGBA: ridimensiona con bordi grigi (letterbox) come ultralytics
  async detect(rgba, width, height) {
    const size = this.size;
    const scale = Math.min(size / width, size / height);
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const padLeft = Math.round((size - newW) / 2 - 0.1);
    const padTop = Math.round((size - newH) / 2 - 0.1);
    const pixels = resizeRgba(rgba, width, height, newW, newH);
    const plane = size * size;
    const input = new Float32Array(3 * plane).fill(PAD_VALUE / 255);
    for (let y = 0; y < newH; y += 1) {
      for (let x = 0; x < newW; x += 1) {
        const s = (y * newW + x) * 4;
        const o = (y + padTop) * size + (x + padLeft);
        input[o] = pixels[s] / 255;
        input[plane + o] = pixels[s + 1] / 255;
        input[2 * plane + o] = pixels[s + 2] / 255;
      }
    }
    const tensor = new this.ort.Tensor("float32", input, [1, 3, size, size]);
    const results = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const output = results[this.session.outputNames[0]];
    return decodeYolo(output.data, output.dims, { conf: this.conf, iou: this.iou, scale, padLeft, padTop, width, height });
  }
}
