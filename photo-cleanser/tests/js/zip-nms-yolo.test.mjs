import assert from "node:assert/strict";
import test from "node:test";
import zlib from "node:zlib";

import { decodeYolo } from "../../frontend/js/logos.js";
import { iou, nms } from "../../frontend/js/nms.js";
import { buildZip, crc32, uniqueName } from "../../frontend/js/zip.js";

test("il CRC-32 coincide con quello di zlib", () => {
  const data = new TextEncoder().encode("Photo Cleanser: foto pulite");
  assert.equal(crc32(data), zlib.crc32(data));
  assert.equal(crc32(new Uint8Array(0)), 0);
});

test("lo ZIP ha una struttura valida e nomi UTF-8", async () => {
  const entries = [
    { name: "foto_redacted.png", data: new Uint8Array([1, 2, 3, 4]) },
    { name: "città.txt", data: new TextEncoder().encode("ciao") },
  ];
  const bytes = Buffer.from(await buildZip(entries).arrayBuffer());
  const end = bytes.length - 22;
  assert.equal(bytes.readUInt32LE(end), 0x06054b50);
  assert.equal(bytes.readUInt16LE(end + 10), 2);
  assert.equal(bytes.readUInt32LE(0), 0x04034b50);
  assert.ok(bytes.includes(Buffer.from("città.txt", "utf8")));
});

test("i nomi duplicati nello ZIP diventano univoci", () => {
  const used = new Set();
  assert.equal(uniqueName("a.png", used), "a.png");
  assert.equal(uniqueName("a.png", used), "a_1.png");
  assert.equal(uniqueName("a.png", used), "a_2.png");
});

test("iou e nms: soglia sul punteggio stretta, soppressione solo sopra la soglia di sovrapposizione", () => {
  assert.equal(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
  assert.equal(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  const boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]];
  const scores = [0.9, 0.8, 0.7, 0.6];
  assert.deepEqual(nms(boxes, scores, { scoreThreshold: 0.6, iouThreshold: 0.3 }), [0, 2]);
  assert.deepEqual(nms(boxes, scores, { scoreThreshold: 0.7, iouThreshold: 0.3 }), [0]);
  assert.deepEqual(nms(boxes, scores, { scoreThreshold: 0.95, iouThreshold: 0.3 }), []);
  assert.deepEqual(nms(boxes, scores, { scoreThreshold: 0, iouThreshold: 0.3, topK: 1 }), [0]);
});

// Crea l'uscita di un YOLO a una classe: [1, 5, N], canale per canale
function yoloOutput(anchors, count) {
  const data = new Float32Array(5 * count);
  anchors.forEach(([cx, cy, w, h, score], i) => {
    data[i] = cx;
    data[count + i] = cy;
    data[2 * count + i] = w;
    data[3 * count + i] = h;
    data[4 * count + i] = score;
  });
  return data;
}

test("YOLO: i riquadri tornano nelle coordinate dell'immagine togliendo bordi e scala del letterbox", () => {
  const data = yoloOutput([[320, 320, 100, 60, 0.9], [322, 321, 100, 60, 0.8], [100, 100, 20, 20, 0.2]], 8);
  // immagine 1280x640 su 640x640: scala 0.5, riempimento verticale di 160 px
  const found = decodeYolo(data, [1, 5, 8], { conf: 0.4, iou: 0.7, scale: 0.5, padLeft: 0, padTop: 160, width: 1280, height: 640 });
  assert.equal(found.length, 1);
  assert.deepEqual(found[0].box, [540 * 1, 260 * 1 - 0, 740, 380].map((v, i) => (i === 1 ? 2 * (320 - 30 - 160) : i === 3 ? 2 * (320 + 30 - 160) : v)));
  assert.equal(found[0].kind, "logo");
  assert.equal(found[0].score, Math.fround(0.9));
});

test("YOLO: la soglia di confidenza scarta i riquadri deboli e il formato end-to-end [1, N, 6] funziona", () => {
  const empty = decodeYolo(yoloOutput([[50, 50, 10, 10, 0.3]], 4), [1, 5, 4], { conf: 0.4, iou: 0.7, scale: 1, padLeft: 0, padTop: 0, width: 100, height: 100 });
  assert.equal(empty.length, 0);
  const e2e = new Float32Array([10, 10, 30, 30, 0.8, 0, 0, 0, 0, 0, 0, 0]);
  const found = decodeYolo(e2e, [1, 2, 6], { conf: 0.4, iou: 0.7, scale: 2, padLeft: 0, padTop: 0, width: 100, height: 100 });
  assert.deepEqual(found.map((d) => d.box), [[5, 5, 15, 15]]);
});
