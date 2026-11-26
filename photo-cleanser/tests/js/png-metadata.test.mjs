import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import zlib from "node:zlib";

import { detectMetadata, sniffFormat } from "../../frontend/js/metadata.js";
import { assertCleanPng, scanPng, stripPng } from "../../frontend/js/png.js";

const fixtures = new URL("../fixtures/", import.meta.url);
const read = (name) => new Uint8Array(fs.readFileSync(new URL(name, fixtures)));

// Costruisce un chunk PNG con CRC corretto
function chunk(type, data) {
  const body = Buffer.concat([Buffer.from(type, "latin1"), Buffer.from(data)]);
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(zlib.crc32(body), 8 + data.length);
  return out;
}

// Costruisce un PNG 1x1 con i chunk extra indicati prima di IDAT
function makePng(extra = [], trailing = Buffer.alloc(0)) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(1, 0);
  header.writeUInt32BE(1, 4);
  header[8] = 8;
  header[9] = 2;
  const pixels = zlib.deflateSync(Buffer.from([0, 255, 0, 0]));
  return new Uint8Array(Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header), ...extra, chunk("IDAT", pixels), chunk("IEND", Buffer.alloc(0)), trailing,
  ]));
}

test("un PNG pulito non ha metadati", () => {
  assert.deepEqual(scanPng(makePng()), []);
  assertCleanPng(makePng());
});

test("vengono riconosciuti EXIF, XMP, IPTC, testo PNG, profilo ICC, C2PA e chunk extra", () => {
  const png = makePng([
    chunk("eXIf", Buffer.from("II*\0")),
    chunk("tEXt", Buffer.from("Author\0Mario")),
    chunk("iTXt", Buffer.from("XML:com.adobe.xmp\0\0\0\0\0<x/>")),
    chunk("zTXt", Buffer.from("Raw profile type iptc\0\0x")),
    chunk("iCCP", Buffer.from("srgb\0\0x")),
    chunk("caBX", Buffer.from("jumb")),
    chunk("pHYs", Buffer.alloc(9)),
  ]);
  assert.deepEqual(scanPng(png).sort(), ["c2pa", "chunk_pHYs", "exif", "icc_profile", "iptc", "png_text", "xmp"]);
  assert.throws(() => assertCleanPng(png), /metadati residui/);
});

test("stripPng toglie tutti i chunk non essenziali e lascia i pixel", () => {
  const dirty = makePng([chunk("tEXt", Buffer.from("A\0b")), chunk("eXIf", Buffer.from("II*\0"))]);
  const clean = stripPng(dirty);
  assert.deepEqual(scanPng(clean), []);
  assert.ok(clean.length < dirty.length);
  const idat = (bytes) => Buffer.from(bytes).toString("latin1").includes("IDAT");
  assert.ok(idat(clean));
});

test("i dati dopo IEND vengono segnalati e rimossi", () => {
  const png = makePng([], Buffer.from("coda nascosta"));
  assert.deepEqual(scanPng(png), ["dati_dopo_IEND"]);
  assert.deepEqual(scanPng(stripPng(png)), []);
});

test("PNG troncato o non valido solleva un errore", () => {
  assert.throws(() => scanPng(makePng().subarray(0, 30)), /troncato|IEND/);
  assert.throws(() => scanPng(new Uint8Array([1, 2, 3])), /non è un PNG/);
});

test("il formato si riconosce dai primi byte, non dall'estensione", () => {
  assert.equal(sniffFormat(read("astronaut.png")), "png");
  assert.equal(sniffFormat(read("astronaut_exif.jpg")), "jpeg");
  assert.equal(sniffFormat(read("non_immagine.png")), null);
  const webp = new Uint8Array(Buffer.from("RIFF\x10\0\0\0WEBPVP8 ", "latin1"));
  assert.equal(sniffFormat(webp), "webp");
});

test("i metadati del JPEG di prova vengono rilevati (EXIF, XMP, commento)", () => {
  const found = detectMetadata(read("astronaut_exif.jpg"));
  for (const kind of ["exif", "xmp", "jpeg_comment"]) assert.ok(found.includes(kind), `${kind} in ${found}`);
  assert.deepEqual(detectMetadata(read("astronaut.png")), []);
});

test("il JPEG con orientamento viene letto come EXIF", () => {
  assert.ok(detectMetadata(read("orientamento_6.jpg")).includes("exif"));
});

test("un WebP con EXIF e XMP viene riconosciuto", () => {
  const riff = (type, size) => Buffer.concat([Buffer.from(type, "latin1"), Buffer.from(new Uint32Array([size]).buffer)]);
  const body = Buffer.concat([Buffer.from("WEBP", "latin1"), riff("VP8 ", 2), Buffer.alloc(2), riff("EXIF", 2), Buffer.alloc(2), riff("XMP ", 2), Buffer.alloc(2)]);
  const file = new Uint8Array(Buffer.concat([Buffer.from("RIFF", "latin1"), Buffer.from(new Uint32Array([body.length]).buffer), body]));
  assert.deepEqual(detectMetadata(file).sort(), ["exif", "xmp"]);
});
