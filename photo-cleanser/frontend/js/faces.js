import { nms } from "./nms.js";
import { resizeRgba } from "./imageops.js";

export const STRIDES = [8, 16, 32];
const DIVISOR = 32;
const f32 = Math.fround;

// Limita un valore tra 0 e 1
function clamp01(value) {
  return Math.min(Math.max(value, 0), 1);
}

// Decodifica le 12 uscite di YuNet 2023mar e applica l'NMS, come fa cv::FaceDetectorYN
export function decodeYuNet(outputs, { inputWidth, inputHeight, scoreThreshold = 0.6, nmsThreshold = 0.3, topK = 5000 }) {
  const padW = Math.ceil(inputWidth / DIVISOR) * DIVISOR;
  const padH = Math.ceil(inputHeight / DIVISOR) * DIVISOR;
  const threshold = f32(scoreThreshold);
  const faces = [];
  for (const stride of STRIDES) {
    const cols = Math.floor(padW / stride);
    const rows = Math.floor(padH / stride);
    const cls = outputs[`cls_${stride}`];
    const obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`];
    const kps = outputs[`kps_${stride}`];
    if (!cls || !obj || !bbox || !kps) {
      throw new Error(`uscite di YuNet mancanti per lo stride ${stride}`);
    }
    if (cls.length !== rows * cols) {
      throw new Error(`griglia inattesa per lo stride ${stride}: ${cls.length} celle invece di ${rows * cols}`);
    }
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const idx = r * cols + c;
        const score = f32(Math.sqrt(f32(clamp01(cls[idx]) * clamp01(obj[idx]))));
        if (!(score > threshold)) {
          continue;
        }
        const cx = f32(f32(c + bbox[idx * 4]) * stride);
        const cy = f32(f32(r + bbox[idx * 4 + 1]) * stride);
        const w = f32(f32(Math.exp(bbox[idx * 4 + 2])) * stride);
        const h = f32(f32(Math.exp(bbox[idx * 4 + 3])) * stride);
        const x1 = f32(cx - f32(w / 2));
        const y1 = f32(cy - f32(h / 2));
        const landmarks = [];
        for (let n = 0; n < 5; n += 1) {
          landmarks.push(f32(f32(kps[idx * 10 + 2 * n] + c) * stride), f32(f32(kps[idx * 10 + 2 * n + 1] + r) * stride));
        }
        faces.push({ x: x1, y: y1, w, h, landmarks, score });
      }
    }
  }
  const rects = faces.map((f) => [Math.trunc(f.x), Math.trunc(f.y), Math.trunc(f.x) + Math.trunc(f.w), Math.trunc(f.y) + Math.trunc(f.h)]);
  const scores = faces.map((f) => f.score);
  const kept = nms(rects, scores, { scoreThreshold: threshold, iouThreshold: f32(nmsThreshold), topK });
  return kept.map((index) => faces[index]);
}

// Rilevatore di volti YuNet eseguito con ONNX Runtime Web
export class YuNetDetector {
  constructor(ort, session, options) {
    this.ort = ort;
    this.session = session;
    this.scoreThreshold = options.scoreThreshold ?? 0.6;
    this.nmsThreshold = options.nmsThreshold ?? 0.3;
    this.maxSide = options.maxSide ?? 2048;
    this.inputName = session.inputNames[0];
    const meta = session.inputMetadata ? session.inputMetadata[0] : null;
    const shape = meta && meta.shape ? meta.shape : [];
    const fixedH = Number.isInteger(shape[2]) && shape[2] > 0 ? shape[2] : null;
    const fixedW = Number.isInteger(shape[3]) && shape[3] > 0 ? shape[3] : null;
    this.fixedSize = fixedH && fixedW ? { width: fixedW, height: fixedH } : null;
  }

  // Crea il rilevatore dal contenuto del file del modello
  static async create(ort, modelBytes, options = {}) {
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    return new YuNetDetector(ort, session, options);
  }

  // Trova i volti in un'immagine RGBA e restituisce riquadri [x1, y1, x2, y2] in pixel dell'immagine
  async detect(rgba, width, height) {
    let scale;
    let target;
    if (this.fixedSize) {
      scale = Math.min(this.fixedSize.width / width, this.fixedSize.height / height);
      target = this.fixedSize;
    } else {
      scale = Math.min(1, this.maxSide / Math.max(width, height));
      target = null;
    }
    const resizedW = Math.max(1, Math.round(width * scale));
    const resizedH = Math.max(1, Math.round(height * scale));
    const pixels = resizeRgba(rgba, width, height, resizedW, resizedH);
    const planeW = target ? target.width : Math.ceil(resizedW / DIVISOR) * DIVISOR;
    const planeH = target ? target.height : Math.ceil(resizedH / DIVISOR) * DIVISOR;
    const input = new Float32Array(3 * planeH * planeW);
    const plane = planeH * planeW;
    for (let y = 0; y < resizedH; y += 1) {
      for (let x = 0; x < resizedW; x += 1) {
        const s = (y * resizedW + x) * 4;
        const o = y * planeW + x;
        input[o] = pixels[s + 2];
        input[plane + o] = pixels[s + 1];
        input[2 * plane + o] = pixels[s];
      }
    }
    const tensor = new this.ort.Tensor("float32", input, [1, 3, planeH, planeW]);
    const results = await this.session.run({ [this.inputName]: tensor });
    const outputs = {};
    for (const name of this.session.outputNames) {
      outputs[name] = results[name].data;
    }
    const faces = decodeYuNet(outputs, {
      inputWidth: target ? target.width : resizedW,
      inputHeight: target ? target.height : resizedH,
      scoreThreshold: this.scoreThreshold,
      nmsThreshold: this.nmsThreshold,
    });
    return faces.map((face) => toDetection(face, scale, width, height));
  }
}

// Converte un volto nelle coordinate dell'immagine originale, con gli stessi arrotondamenti della versione Python
function toDetection(face, scale, width, height) {
  const x1 = Math.max(0, Math.floor(face.x / scale));
  const y1 = Math.max(0, Math.floor(face.y / scale));
  const x2 = Math.min(width, Math.ceil((face.x + face.w) / scale));
  const y2 = Math.min(height, Math.ceil((face.y + face.h) / scale));
  return { kind: "volto", box: [x1, y1, x2, y2], score: face.score, landmarks: face.landmarks.map((v) => v / scale) };
}
