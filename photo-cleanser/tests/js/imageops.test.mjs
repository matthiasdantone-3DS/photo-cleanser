import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import { resizeArea, resizeBilinear, resizeRgba } from "../../frontend/js/imageops.js";

const golden = JSON.parse(fs.readFileSync(new URL("./fixtures/resize_golden.json", import.meta.url), "utf8"));

// Differenza massima tra due array di byte
function maxDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

test("la riduzione con media delle aree coincide con cv2.INTER_AREA (al massimo 1 livello di scarto)", () => {
  for (const item of golden.cases.filter((c) => c.kind === "area")) {
    const out = resizeArea(Uint8ClampedArray.from(item.src), item.sw, item.sh, item.dw, item.dh);
    assert.ok(maxDiff(out, item.expected) <= 1, `${item.sw}x${item.sh} -> ${item.dw}x${item.dh}`);
  }
});

test("l'ingrandimento bilineare coincide con cv2.INTER_LINEAR (al massimo 2 livelli di scarto)", () => {
  for (const item of golden.cases.filter((c) => c.kind === "bilinear")) {
    const out = resizeBilinear(Uint8ClampedArray.from(item.src), item.sw, item.sh, item.dw, item.dh);
    assert.ok(maxDiff(out, item.expected) <= 2, `${item.sw}x${item.sh} -> ${item.dw}x${item.dh}`);
  }
});

test("resizeRgba sceglie il metodo giusto e non copia se la dimensione è uguale", () => {
  const src = new Uint8ClampedArray(4 * 4 * 4).fill(200);
  assert.equal(resizeRgba(src, 4, 4, 4, 4), src);
  assert.equal(resizeRgba(src, 4, 4, 2, 2).length, 2 * 2 * 4);
  assert.equal(resizeRgba(src, 4, 4, 8, 8).length, 8 * 8 * 4);
});
