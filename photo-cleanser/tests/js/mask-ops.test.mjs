import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import { buildSubjectMask, centerInsideMask, thresholdUpsample } from "../../frontend/js/mask-ops.js";

const golden = JSON.parse(fs.readFileSync(new URL("./fixtures/mask_golden.json", import.meta.url), "utf8"));

// Converte una stringa di 0 e 1 in un array di byte
function toBytes(text) {
  return Uint8Array.from(text, (ch) => (ch === "1" ? 1 : 0));
}

test("la maschera coincide con la pipeline Python/OpenCV (chiusura 5x5 e filtro 2%)", () => {
  for (const [index, item] of golden.cases.entries()) {
    const result = buildSubjectMask(toBytes(item.input), item.w, item.h, { morphKernel: 5, minComponentRatio: 0.02 });
    assert.equal(result.empty, item.empty, `caso ${index}: stato vuoto`);
    assert.deepEqual(Array.from(result.mask), Array.from(toBytes(item.expected)), `caso ${index}: maschera`);
  }
});

test("una maschera vuota viene segnalata e non viene modificata", () => {
  const result = buildSubjectMask(new Uint8Array(30), 6, 5);
  assert.equal(result.empty, true);
});

test("l'ingrandimento bilineare con soglia mantiene la forma", () => {
  const prob = new Float32Array(4 * 4);
  for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) prob[y * 4 + x] = x >= 2 ? 1 : 0;
  const mask = thresholdUpsample(prob, 4, 4, 8, 8, 0.5);
  assert.equal(mask[0], 0);
  assert.equal(mask[7], 1);
  assert.equal(mask[7 * 8 + 7], 1);
});

test("il centro di un riquadro viene controllato dentro la maschera", () => {
  const mask = new Uint8Array(10 * 10);
  mask[5 * 10 + 5] = 1;
  assert.equal(centerInsideMask([4, 4, 7, 7], mask, 10, 10), true);
  assert.equal(centerInsideMask([0, 0, 3, 3], mask, 10, 10), false);
});
