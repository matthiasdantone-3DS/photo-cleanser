import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build as esbuild } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const strict = process.env.CI === "true" || process.argv.includes("--strict");

// Checksum ufficiale di face_detection_yunet_2023mar.onnx (puntatore Git LFS del repository OpenCV Zoo)
const YUNET = {
  file: "face_detection_yunet_2023mar.onnx",
  url: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
  sha256: process.env.YUNET_SHA256 || "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
  bytes: 232589,
};

// Calcola lo SHA-256 di un buffer
function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

// Copia un file creando le cartelle necessarie
function copy(from, to) {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
}

// Scarica YuNet dal repository OpenCV Zoo e verifica il checksum; restituisce null se non riesce
async function downloadYunet() {
  try {
    const response = await fetch(YUNET.url, { redirect: "follow" });
    if (!response.ok) {
      throw new Error(`risposta ${response.status}`);
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    const actual = sha256(buffer);
    if (actual !== YUNET.sha256) {
      throw new Error(`checksum diverso: atteso ${YUNET.sha256}, ottenuto ${actual} (${buffer.length} byte)`);
    }
    return buffer;
  } catch (error) {
    console.warn(`YuNet: download non riuscito (${error.message})`);
    return null;
  }
}

// Inserisce un modello in dist/model e lo descrive nel manifest
function addModel(manifest, key, buffer, name, source) {
  fs.mkdirSync(path.join(dist, "model"), { recursive: true });
  fs.writeFileSync(path.join(dist, "model", name), buffer);
  manifest.models[key] = { file: `model/${name}`, sha256: sha256(buffer), bytes: buffer.length, source };
}

fs.rmSync(dist, { recursive: true, force: true });
fs.cpSync(path.join(root, "frontend"), dist, { recursive: true });
fs.writeFileSync(path.join(dist, ".nojekyll"), "");

// ONNX Runtime Web: solo la versione WebAssembly, servita dal sito stesso
const ortDir = path.join(root, "node_modules", "onnxruntime-web", "dist");
for (const name of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copy(path.join(ortDir, name), path.join(dist, "vendor", "ort", name));
}

// LiteRT.js: il codice viene impacchettato in un solo file e i runtime WebAssembly vengono copiati
const liteDir = path.join(root, "node_modules", "@litertjs", "core");
await esbuild({
  entryPoints: [path.join(liteDir, "dist", "index.js")],
  bundle: true,
  format: "esm",
  outfile: path.join(dist, "vendor", "litert", "litert.js"),
  logLevel: "warning",
});
for (const name of ["litert_wasm_internal", "litert_wasm_compat_internal"]) {
  for (const extension of [".js", ".wasm"]) {
    copy(path.join(liteDir, "wasm", name + extension), path.join(dist, "vendor", "litert", name + extension));
  }
}

// Il worker è uno script classico: LiteRT.js usa importScripts, che i worker di tipo modulo non permettono
await esbuild({
  entryPoints: [path.join(root, "frontend", "js", "worker.js")],
  bundle: true,
  format: "iife",
  outfile: path.join(dist, "js", "worker.bundle.js"),
  logLevel: "warning",
});
fs.rmSync(path.join(dist, "js", "worker.js"));

const manifest = {
  version: process.env.GITHUB_SHA ? process.env.GITHUB_SHA.slice(0, 7) : "locale",
  built_at: new Date().toISOString(),
  models: { segmentation: null, faces: null, logos: null },
};

// Segmentazione: il file TFLite originale usato dall'algoritmo Python (MediaPipe Selfie Segmentation)
const segmentation = fs.readFileSync(path.join(root, "model", "selfie_segmentation.tflite"));
addModel(manifest, "segmentation", segmentation, "selfie_segmentation.tflite", "MediaPipe Selfie Segmentation (modello originale, nessuna conversione)");

// Volti: file nel repository oppure scaricato da OpenCV Zoo; in entrambi i casi controllato con lo SHA-256 ufficiale
const localYunet = path.join(root, "model", YUNET.file);
let yunet = fs.existsSync(localYunet) ? fs.readFileSync(localYunet) : null;
if (yunet && sha256(yunet) !== YUNET.sha256) {
  throw new Error(`model/${YUNET.file} non coincide con il checksum ufficiale (${YUNET.sha256})`);
}
yunet = yunet || (await downloadYunet());
if (yunet) {
  addModel(manifest, "faces", yunet, YUNET.file, "YuNet 2023mar da OpenCV Zoo (modello originale)");
} else if (strict) {
  throw new Error("YuNet non disponibile: aggiungi model/" + YUNET.file + " al repository (vedi model/README.md)");
} else {
  console.warn("ATTENZIONE: il sito verrà costruito senza il rilevatore di volti.");
}

// Loghi: modello ONNX esportato da training/export_onnx.py, facoltativo
const logoPath = path.join(root, "model", "logo.onnx");
if (fs.existsSync(logoPath)) {
  addModel(manifest, "logos", fs.readFileSync(logoPath), "logo.onnx", "YOLO addestrato con training/ ed esportato in ONNX");
}

fs.copyFileSync(path.join(root, "THIRD_PARTY_NOTICES.md"), path.join(dist, "THIRD_PARTY_NOTICES.md"));
fs.writeFileSync(path.join(dist, "model-manifest.json"), JSON.stringify(manifest, null, 2));
console.log("sito costruito in dist/");
for (const [key, entry] of Object.entries(manifest.models)) {
  console.log(`  ${key}: ${entry ? `${entry.file} (${entry.bytes} byte)` : "non presente"}`);
}
