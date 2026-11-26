const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const KEEP = new Set(["IHDR", "PLTE", "IDAT", "IEND"]);

// Legge un intero a 32 bit big endian
function readU32(bytes, offset) {
  return ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0;
}

// Controlla la firma iniziale di un PNG
export function isPng(bytes) {
  return bytes.length >= 8 && SIGNATURE.every((value, index) => bytes[index] === value);
}

// Scorre i chunk di un PNG e li passa alla funzione visit(tipo, inizio, fine); segnala dati dopo IEND
function walk(bytes, visit) {
  if (!isPng(bytes)) {
    throw new Error("il file non è un PNG valido");
  }
  let pos = 8;
  while (pos + 12 <= bytes.length) {
    const length = readU32(bytes, pos);
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    const end = pos + 12 + length;
    if (end > bytes.length) {
      throw new Error("chunk PNG troncato");
    }
    visit(type, pos, end);
    pos = end;
    if (type === "IEND") {
      return pos < bytes.length;
    }
  }
  throw new Error("PNG senza chunk IEND");
}

// Associa un chunk PNG al tipo di metadato che rappresenta (null se è un chunk ammesso)
export function classifyChunk(type, bytes, start, end) {
  if (KEEP.has(type)) {
    return null;
  }
  if (type === "eXIf") return "exif";
  if (type === "iCCP") return "icc_profile";
  if (type === "caBX") return "c2pa";
  if (type === "tEXt" || type === "zTXt" || type === "iTXt") {
    let keyword = "";
    for (let i = start + 8; i < end - 4 && bytes[i] !== 0; i += 1) {
      keyword += String.fromCharCode(bytes[i]).toLowerCase();
    }
    if (keyword === "xml:com.adobe.xmp") return "xmp";
    if (keyword.includes("iptc")) return "iptc";
    if (keyword.includes("exif")) return "exif";
    return "png_text";
  }
  return `chunk_${type}`;
}

// Elenca i metadati presenti in un PNG (lista vuota se è pulito)
export function scanPng(bytes) {
  const found = [];
  const trailing = walk(bytes, (type, start, end) => {
    const marker = classifyChunk(type, bytes, start, end);
    if (marker && !found.includes(marker)) {
      found.push(marker);
    }
  });
  if (trailing) {
    found.push("dati_dopo_IEND");
  }
  return found;
}

// Ricostruisce il PNG tenendo solo i chunk indispensabili (intestazione, palette, pixel, fine)
export function stripPng(bytes) {
  const parts = [bytes.subarray(0, 8)];
  walk(bytes, (type, start, end) => {
    if (KEEP.has(type)) {
      parts.push(bytes.subarray(start, end));
    }
  });
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

// Solleva un errore se il PNG contiene metadati
export function assertCleanPng(bytes) {
  const leftovers = scanPng(bytes);
  if (leftovers.length) {
    throw new Error(`metadati residui nel PNG: ${leftovers.join(", ")}`);
  }
}
