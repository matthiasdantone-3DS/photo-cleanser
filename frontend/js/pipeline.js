import { LIMITS, SETTINGS } from "./config.js";
import { decodeImage, encodePng } from "./image-io.js";
import { centerInsideMask } from "./mask-ops.js";
import { assertCleanPng, stripPng } from "./png.js";
import { redactInPlace } from "./redact.js";

export const STATUS = { OK: "ok", NO_SUBJECT: "nessun_soggetto", REVIEW: "da_rivedere", ERROR: "errore" };

// Pipeline completa nel browser: lettura, segmentazione, rilevamento, oscuramento, verifica, PNG pulito
export class RedactionPipeline {
  constructor({ segmenter, faceDetector, logoDetector, phaseErrors = {}, segmenterError = null, settings = SETTINGS, limits = LIMITS, redactBackground = false }) {
    this.segmenter = segmenter;
    this.faceDetector = faceDetector;
    this.logoDetector = logoDetector;
    this.phaseErrors = phaseErrors;
    this.segmenterError = segmenterError;
    this.settings = settings;
    this.limits = limits;
    // Di default lo sfondo resta com'è: si oscurano solo volti e loghi
    this.redactBackground = redactBackground === true;
  }

  // Elabora un file e restituisce sempre un risultato (anche in caso di errore) con il PNG finale se disponibile
  async process(file) {
    const started = performance.now();
    const result = {
      input: file.name,
      status: STATUS.OK,
      detections: { volto: 0, logo: 0 },
      detectionDetails: [],
      backgroundRedactedPx: 0,
      metadataRemoved: [],
      errors: [],
      width: 0,
      height: 0,
      originalWidth: 0,
      originalHeight: 0,
      downscaled: false,
      elapsedMs: 0,
    };
    let blob = null;
    try {
      blob = await this.run(file, result);
    } catch (error) {
      result.status = STATUS.ERROR;
      result.errors.push(error && error.message ? error.message : String(error));
      blob = null;
    }
    result.elapsedMs = Math.round(performance.now() - started);
    return { result, blob };
  }

  // Passi dell'algoritmo, nello stesso ordine della versione Python
  async run(file, result) {
    const image = await decodeImage(file, this.limits);
    const { rgba, width, height } = image;
    result.metadataRemoved = image.metadata;
    Object.assign(result, {
      width,
      height,
      originalWidth: image.originalWidth,
      originalHeight: image.originalHeight,
      downscaled: image.downscaled,
    });
    if (!this.segmenter) {
      throw new Error(this.segmenterError || "segmentazione non disponibile");
    }
    const subject = await this.segmenter.segment(rgba, width, height);
    if (subject.empty) {
      result.status = STATUS.NO_SUBJECT;
      return encodePng(rgba, width, height, stripPng, assertCleanPng);
    }
    const redactions = [];
    for (const face of await this.runDetector("volto", this.faceDetector, image, result)) {
      const inside = centerInsideMask(face.box, subject.mask, width, height);
      this.record(result, face, !inside);
      if (!inside) {
        redactions.push(face.box);
      }
    }
    for (const logo of await this.runDetector("logo", this.logoDetector, image, result)) {
      this.record(result, logo, true);
      redactions.push(logo.box);
    }
    // Lo sfondo viene passato solo se l'opzione è attiva; altrimenti la maschera è null e i pixel restano invariati
    const backgroundMask = this.redactBackground ? subject.mask : null;
    result.backgroundRedactedPx = redactInPlace(rgba, width, height, backgroundMask, redactions, this.settings.padPx);
    const blob = await encodePng(rgba, width, height, stripPng, assertCleanPng);
    await this.verify(rgba, width, height, subject.mask, result);
    return blob;
  }

  // Lancia un rilevatore: un suo errore viene registrato e la fase saltata
  async runDetector(kind, detector, image, result) {
    if (!detector) {
      if (this.phaseErrors[kind]) {
        this.flagReview(result, `${kind}: fase saltata (${this.phaseErrors[kind]})`);
      }
      return [];
    }
    try {
      return await detector.detect(image.rgba, image.width, image.height);
    } catch (error) {
      this.flagReview(result, `${kind}: errore del rilevatore (${error.message})`);
      return [];
    }
  }

  // Annota un rilevamento e aggiorna i conteggi di quelli oscurati
  record(result, detection, redacted) {
    result.detectionDetails.push({ kind: detection.kind, box: detection.box, score: Math.round(detection.score * 10000) / 10000, redacted });
    if (redacted) {
      result.detections[detection.kind] += 1;
    }
  }

  // Segna il file come da rivedere e registra il motivo
  flagReview(result, message) {
    result.errors.push(message);
    if (result.status === STATUS.OK) {
      result.status = STATUS.REVIEW;
    }
  }

  // Rilancia il rilevatore di volti sull'immagine già oscurata: ogni volto fuori dal soggetto va rivisto
  async verify(rgba, width, height, mask, result) {
    if (!this.faceDetector) {
      return;
    }
    try {
      const found = await this.faceDetector.detect(rgba, width, height);
      const leftovers = found.filter((face) => !centerInsideMask(face.box, mask, width, height));
      if (leftovers.length) {
        this.flagReview(
          result,
          leftovers.length === 1
            ? "1 volto non coperto è ancora rilevabile nell'output"
            : `${leftovers.length} volti non coperti sono ancora rilevabili nell'output`
        );
      }
    } catch (error) {
      this.flagReview(result, `verifica finale dei volti non riuscita (${error.message})`);
    }
  }
}
