import { LIMITS } from "./config.js";
import { detectMetadata, sniffFormat } from "./metadata.js";

// Crea una superficie di disegno, anche nei worker
function makeCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") {
    return new OffscreenCanvas(width, height);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

// Legge un file immagine: valida formato e dimensioni, applica l'orientamento EXIF e compone su fondo bianco
export async function decodeImage(file, limits = LIMITS) {
  if (file.size > limits.maxFileBytes) {
    throw new Error(`il file supera il limite di ${Math.round(limits.maxFileBytes / 1048576)} MB`);
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  const format = sniffFormat(bytes);
  if (!format) {
    throw new Error("il file non è un'immagine JPEG, PNG o WebP");
  }
  const metadata = detectMetadata(bytes);
  let bitmap;
  try {
    bitmap = await createImageBitmap(new Blob([bytes], { type: `image/${format}` }), { imageOrientation: "from-image" });
  } catch (error) {
    throw new Error("impossibile leggere l'immagine: il file potrebbe essere danneggiato");
  }
  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;
  if (originalWidth * originalHeight > limits.maxDecodePixels) {
    bitmap.close();
    throw new Error(`immagine troppo grande: ${originalWidth}x${originalHeight} pixel`);
  }
  const scale = Math.min(1, limits.maxSide / Math.max(originalWidth, originalHeight), Math.sqrt(limits.maxPixels / (originalWidth * originalHeight)));
  const width = Math.max(1, Math.round(originalWidth * scale));
  const height = Math.max(1, Math.round(originalHeight * scale));
  const canvas = makeCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  return { rgba, width, height, originalWidth, originalHeight, downscaled: width !== originalWidth, metadata, format };
}

// Codifica i pixel come PNG: il browser scrive il file e poi tutti i chunk non essenziali vengono tolti
export async function encodePng(rgba, width, height, stripPng, assertCleanPng) {
  const canvas = makeCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = canvas.convertToBlob
    ? await canvas.convertToBlob({ type: "image/png" })
    : await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const clean = stripPng(new Uint8Array(await blob.arrayBuffer()));
  assertCleanPng(clean);
  return new Blob([clean], { type: "image/png" });
}
