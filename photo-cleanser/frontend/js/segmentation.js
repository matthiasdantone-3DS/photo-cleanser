import { buildSubjectMask, thresholdUpsample } from "./mask-ops.js";

export const MODEL_SIZE = 256;

// Prepara l'ingresso del modello: 256x256, valori tra 0 e 1, bilineare con bordi neri come il grafo MediaPipe originale
export function toModelInput(rgba, width, height) {
  const input = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
  const scaleX = width / MODEL_SIZE;
  const scaleY = height / MODEL_SIZE;
  const at = (x, y, c) => (x < 0 || y < 0 || x >= width || y >= height ? 0 : rgba[(y * width + x) * 4 + c]);
  for (let y = 0; y < MODEL_SIZE; y += 1) {
    const fy = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    for (let x = 0; x < MODEL_SIZE; x += 1) {
      const fx = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.floor(fx);
      const wx = fx - x0;
      for (let c = 0; c < 3; c += 1) {
        const value =
          (at(x0, y0, c) * (1 - wx) + at(x0 + 1, y0, c) * wx) * (1 - wy) +
          (at(x0, y0 + 1, c) * (1 - wx) + at(x0 + 1, y0 + 1, c) * wx) * wy;
        input[(y * MODEL_SIZE + x) * 3 + c] = Math.round(value) / 255;
      }
    }
  }
  return input;
}

// Segmentazione persona con il modello originale MediaPipe Selfie Segmentation (selfie_segmentation.tflite) eseguito da LiteRT.js
export class SubjectSegmenter {
  constructor(litert, model, settings) {
    this.litert = litert;
    this.model = model;
    this.settings = settings;
  }

  // Carica il runtime LiteRT e compila il modello dal suo contenuto
  static async create(litert, wasmDirUrl, modelBytes, settings) {
    // In un worker Emscripten cerca il file .wasm accanto al worker: lo indichiamo noi
    if (typeof self !== "undefined" && typeof self.importScripts === "function") {
      self.Module = { locateFile: (name) => new URL(name, wasmDirUrl).href };
    }
    await litert.loadLiteRt(wasmDirUrl);
    const model = await litert.loadAndCompile(modelBytes, { accelerator: "wasm" });
    return new SubjectSegmenter(litert, model, settings);
  }

  // Restituisce la maschera del soggetto (0/1 per pixel) per un'immagine RGBA
  async segment(rgba, width, height) {
    const input = new this.litert.Tensor(toModelInput(rgba, width, height), [1, MODEL_SIZE, MODEL_SIZE, 3]);
    let probabilities;
    const outputs = await this.model.run(input);
    const first = await outputs[0].moveTo("wasm");
    probabilities = Float32Array.from(first.toTypedArray());
    first.delete();
    if (typeof outputs.delete === "function") {
      outputs.delete();
    }
    input.delete();
    const binary = thresholdUpsample(probabilities, MODEL_SIZE, MODEL_SIZE, width, height, this.settings.maskThreshold);
    return buildSubjectMask(binary, width, height, {
      morphKernel: this.settings.morphKernel,
      minComponentRatio: this.settings.maskMinComponentRatio,
    });
  }
}
