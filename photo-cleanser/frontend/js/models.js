import { SETTINGS } from "./config.js";
import { LogoDetector } from "./logos.js";
import { YuNetDetector } from "./faces.js";
import { RedactionPipeline } from "./pipeline.js";
import { SubjectSegmenter } from "./segmentation.js";

// Calcola lo SHA-256 di un file e lo restituisce in esadecimale
export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

// Scarica un file del sito e controlla lo SHA-256 indicato nel manifest
async function fetchModel(base, entry, label) {
  const response = await fetch(new URL(entry.file, base), { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`${label}: download non riuscito (${response.status})`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (entry.sha256) {
    const actual = await sha256Hex(bytes);
    if (actual !== entry.sha256) {
      throw new Error(`${label}: controllo di integrità fallito (checksum diverso da quello atteso)`);
    }
  }
  return bytes;
}

// Carica ONNX Runtime Web dal sito stesso (nessuna CDN) in modalità WebAssembly a un solo thread
async function loadOrt(base) {
  const ort = await import(new URL("vendor/ort/ort.wasm.min.mjs", base).href);
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", base).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  return ort;
}

// Prepara tutto il necessario: manifest, modelli, runtime. Una fase che fallisce viene saltata e segnalata
export async function createEngine(onProgress, base) {
  const manifestResponse = await fetch(new URL("model-manifest.json", base), { cache: "no-cache" });
  if (!manifestResponse.ok) {
    throw new Error("manifest dei modelli non trovato");
  }
  const manifest = await manifestResponse.json();
  const phaseErrors = {};
  const skipped = [];
  let segmenter = null;
  let segmenterError = null;
  let faceDetector = null;
  let logoDetector = null;
  let ort = null;

  onProgress({ stage: "segmentation", message: "Preparo il modello di segmentazione" });
  try {
    const bytes = await fetchModel(base, manifest.models.segmentation, "segmentazione");
    const litert = await import(new URL("vendor/litert/litert.js", base).href);
    segmenter = await SubjectSegmenter.create(litert, new URL("vendor/litert/", base).href, bytes, SETTINGS);
  } catch (error) {
    segmenterError = `segmentazione non disponibile: ${error.message}`;
  }

  const faces = manifest.models.faces;
  if (faces) {
    onProgress({ stage: "faces", message: "Preparo il rilevatore di volti" });
    try {
      const bytes = await fetchModel(base, faces, "volti");
      ort = ort || (await loadOrt(base));
      faceDetector = await YuNetDetector.create(ort, bytes, { scoreThreshold: SETTINGS.faceScore });
    } catch (error) {
      phaseErrors.volto = error.message;
      skipped.push({ phase: "volto", reason: error.message });
    }
  } else {
    phaseErrors.volto = "modello YuNet non presente nel sito";
    skipped.push({ phase: "volto", reason: phaseErrors.volto });
  }

  const logos = manifest.models.logos;
  if (logos) {
    onProgress({ stage: "logos", message: "Preparo il rilevatore di loghi" });
    try {
      const bytes = await fetchModel(base, logos, "loghi");
      ort = ort || (await loadOrt(base));
      logoDetector = await LogoDetector.create(ort, bytes, { conf: SETTINGS.logoConf });
    } catch (error) {
      phaseErrors.logo = error.message;
      skipped.push({ phase: "logo", reason: error.message });
    }
  } else {
    skipped.push({ phase: "logo", reason: "nessun modello di loghi presente (vedi training/README.md)" });
  }

  const pipeline = new RedactionPipeline({ segmenter, faceDetector, logoDetector, phaseErrors, segmenterError });
  return {
    pipeline,
    capabilities: {
      segmentation: Boolean(segmenter),
      faces: Boolean(faceDetector),
      logos: Boolean(logoDetector),
      skipped,
      segmenterError,
      models: manifest.models,
    },
  };
}
