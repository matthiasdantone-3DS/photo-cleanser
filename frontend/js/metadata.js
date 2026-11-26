import { classifyChunk, isPng } from "./png.js";

// Cerca una sequenza di caratteri ASCII in un array di byte
function includesAscii(bytes, text, from = 0, to = bytes.length) {
  const needle = Array.from(text, (ch) => ch.charCodeAt(0));
  outer: for (let i = from; i + needle.length <= to; i += 1) {
    for (let j = 0; j < needle.length; j += 1) {
      if (bytes[i + j] !== needle[j]) {
        continue outer;
      }
    }
    return true;
  }
  return false;
}

// Riconosce il formato dai primi byte del file: "jpeg", "png", "webp" oppure null
export function sniffFormat(bytes) {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpeg";
  if (isPng(bytes)) return "png";
  if (bytes.length >= 12 && includesAscii(bytes, "RIFF", 0, 4) && includesAscii(bytes, "WEBP", 8, 12)) return "webp";
  return null;
}

// Aggiunge un tipo di metadato senza duplicati
function add(found, kind) {
  if (!found.includes(kind)) {
    found.push(kind);
  }
}

// Scorre i segmenti di un JPEG fino ai dati dell'immagine
function scanJpeg(bytes, found) {
  let pos = 2;
  while (pos + 4 <= bytes.length && bytes[pos] === 0xff) {
    const marker = bytes[pos + 1];
    if (marker === 0xda || marker === 0xd9) break;
    if (marker === 0xff) {
      pos += 1;
      continue;
    }
    const length = (bytes[pos + 2] << 8) | bytes[pos + 3];
    const start = pos + 4;
    const end = Math.min(bytes.length, pos + 2 + length);
    if (marker === 0xe1) {
      if (includesAscii(bytes, "Exif", start, Math.min(end, start + 6))) add(found, "exif");
      else if (includesAscii(bytes, "http://ns.adobe.com/xap", start, Math.min(end, start + 32))) add(found, "xmp");
    } else if (marker === 0xe2 && includesAscii(bytes, "ICC_PROFILE", start, Math.min(end, start + 16))) {
      add(found, "icc_profile");
    } else if (marker === 0xed) {
      add(found, "iptc");
    } else if (marker === 0xfe) {
      add(found, "jpeg_comment");
    }
    pos = end;
  }
}

// Scorre i chunk di un file WebP (RIFF)
function scanWebp(bytes, found) {
  let pos = 12;
  while (pos + 8 <= bytes.length) {
    const type = String.fromCharCode(bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]);
    const size = bytes[pos + 4] | (bytes[pos + 5] << 8) | (bytes[pos + 6] << 16) | (bytes[pos + 7] << 24);
    if (type === "EXIF") add(found, "exif");
    if (type === "XMP ") add(found, "xmp");
    if (type === "ICCP") add(found, "icc_profile");
    pos += 8 + size + (size % 2);
  }
}

// Elenca i tipi di metadato presenti nel file originale (la ricodifica li elimina tutti)
export function detectMetadata(bytes) {
  const found = [];
  const format = sniffFormat(bytes);
  if (format === "jpeg") scanJpeg(bytes, found);
  if (format === "webp") scanWebp(bytes, found);
  if (format === "png") {
    let pos = 8;
    while (pos + 12 <= bytes.length) {
      const length = ((bytes[pos] << 24) | (bytes[pos + 1] << 16) | (bytes[pos + 2] << 8) | bytes[pos + 3]) >>> 0;
      const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
      const end = pos + 12 + length;
      if (end > bytes.length) break;
      const marker = classifyChunk(type, bytes, pos, end);
      if (marker) add(found, marker);
      pos = end;
      if (type === "IEND") break;
    }
  }
  if (format !== "png" && includesAscii(bytes, "jumb") && includesAscii(bytes, "c2pa")) add(found, "c2pa");
  return found;
}
