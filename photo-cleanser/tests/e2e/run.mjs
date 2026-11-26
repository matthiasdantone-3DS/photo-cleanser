import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

import { chromium } from "playwright-core";

import { scanPng } from "../../frontend/js/png.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const dist = path.join(root, "dist");
const fixtures = path.join(root, "tests", "fixtures");
const resultsDir = path.join(root, "test-results");
const BASE_PATH = "/photo-cleanser/";
const SYNTHETIC = process.env.E2E_SYNTHETIC_FACES === "1";
const PROCESS_TIMEOUT = 180000;
const PAD_PX = 6;
const DONE = '.item[data-state="ok"], .item[data-state="da_rivedere"], .item[data-state="nessun_soggetto"], .item[data-state="errore"]';

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".tflite": "application/octet-stream", ".md": "text/markdown" };
const requestLog = [];

fs.mkdirSync(resultsDir, { recursive: true });

// Server statico che imita GitHub Pages: il sito vive sotto BASE_PATH e non ha intestazioni speciali
function startServer() {
  const server = http.createServer((req, res) => {
    requestLog.push({ method: req.method, url: req.url });
    const url = new URL(req.url, "http://localhost");
    if (url.pathname.startsWith("/fixtures/")) {
      const fixture = path.join(fixtures, decodeURIComponent(url.pathname.slice("/fixtures/".length)));
      if (!fixture.startsWith(fixtures) || !fs.existsSync(fixture)) {
        res.writeHead(404).end("non trovato");
        return;
      }
      res.writeHead(200, { "Content-Type": "application/octet-stream" });
      fs.createReadStream(fixture).pipe(res);
      return;
    }
    if (!url.pathname.startsWith(BASE_PATH)) {
      res.writeHead(404).end("non trovato");
      return;
    }
    let relative = decodeURIComponent(url.pathname.slice(BASE_PATH.length)) || "index.html";
    if (relative.endsWith("/")) relative += "index.html";
    const file = path.join(dist, relative);
    if (!file.startsWith(dist) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("non trovato");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream", "Cache-Control": "no-cache" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

// In modalità sintetica sostituisce il file YuNet con un modello di prova a volti noti e aggiorna il manifest
function installSyntheticFaces() {
  const model = fs.readFileSync(path.join(fixtures, "yunet_sintetico_e2e.onnx"));
  fs.mkdirSync(path.join(dist, "model"), { recursive: true });
  fs.writeFileSync(path.join(dist, "model", "face_detection_yunet_2023mar.onnx"), model);
  const manifestPath = path.join(dist, "model-manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  manifest.models.faces = {
    file: "model/face_detection_yunet_2023mar.onnx",
    sha256: createHash("sha256").update(model).digest("hex"),
    bytes: model.length,
    source: "SINTETICO: solo per i test",
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

// Decodifica la maschera di riferimento (run-length) prodotta dall'algoritmo Python
function loadGoldenMask() {
  const golden = JSON.parse(fs.readFileSync(path.join(fixtures, "astronaut_mask_golden.json"), "utf8"));
  const mask = new Uint8Array(golden.width * golden.height);
  let value = golden.first;
  let pos = 0;
  for (const run of golden.runs) {
    mask.fill(value, pos, pos + run);
    pos += run;
    value ^= 1;
  }
  return { width: golden.width, height: golden.height, mask: Array.from(mask).join("") };
}

// Riquadri oscurati (espansi del margine) come li calcola la pipeline
function redactedBoxes(details, width, height) {
  return details
    .filter((d) => d.redacted)
    .map((d) => [Math.max(0, d.box[0] - PAD_PX), Math.max(0, d.box[1] - PAD_PX), Math.min(width, d.box[2] + PAD_PX), Math.min(height, d.box[3] + PAD_PX)]);
}

const results = [];

// Esegue una prova e ne registra l'esito
async function check(name, fn) {
  const started = Date.now();
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`  ok  ${name} (${Math.round((Date.now() - started) / 100) / 10} s)`);
  } catch (error) {
    results.push({ name, ok: false, error });
    console.log(`  KO  ${name}\n      ${String(error.message).split("\n").join("\n      ")}`);
  }
}

// Apre la pagina e raccoglie errori di console, violazioni CSP e richieste di rete
async function openPage(context, origin) {
  const page = await context.newPage();
  page.problems = [];
  page.requests = [];
  page.on("pageerror", (error) => page.problems.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    const text = message.text();
    if (/Content Security Policy|Refused to|violates/i.test(text)) page.problems.push(`CSP: ${text}`);
  });
  page.on("request", (request) => page.requests.push({ method: request.method(), url: request.url(), body: request.postData() }));
  await page.goto(`${origin}${BASE_PATH}`);
  await page.waitForSelector('body[data-engine="ready"], body[data-engine="error"]', { timeout: PROCESS_TIMEOUT });
  return page;
}

// Aspetta che la scheda in posizione index sia terminata
async function waitDone(page, index) {
  await page.waitForFunction(
    ([i, selector]) => {
      const card = document.querySelectorAll(".item")[i];
      return Boolean(card) && card.matches(selector);
    },
    [index, DONE],
    { timeout: PROCESS_TIMEOUT }
  );
}

// Carica un file di prova e aspetta che la scheda sia finita; restituisce stato, dati e posizione della scheda
async function upload(page, fileName) {
  const before = await page.locator(".item").count();
  await page.setInputFiles("#file-input", path.join(fixtures, fileName));
  const card = page.locator(".item").nth(before);
  await card.waitFor({ timeout: 10000 });
  await waitDone(page, before);
  const state = await card.getAttribute("data-state");
  const result = JSON.parse((await card.getAttribute("data-result")) || "null");
  return { card, state, result };
}

// Scarica il PNG di una scheda e lo restituisce come buffer
async function downloadPng(page, card) {
  const [download] = await Promise.all([page.waitForEvent("download"), card.getByRole("button", { name: "Scarica PNG" }).click()]);
  return fs.readFileSync(await download.path());
}

// Legge larghezza e altezza dall'intestazione di un PNG
function pngSize(bytes) {
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

// Legge i chunk di un file ZIP e restituisce nome e contenuto di ogni voce
function readZip(buffer) {
  const end = buffer.length - 22;
  assert.equal(buffer.readUInt32LE(end), 0x06054b50, "fine dell'archivio ZIP non trovata");
  const count = buffer.readUInt16LE(end + 10);
  let pos = buffer.readUInt32LE(end + 16);
  const entries = [];
  for (let i = 0; i < count; i += 1) {
    assert.equal(buffer.readUInt32LE(pos), 0x02014b50);
    const crc = buffer.readUInt32LE(pos + 16);
    const size = buffer.readUInt32LE(pos + 24);
    const nameLength = buffer.readUInt16LE(pos + 28);
    const localOffset = buffer.readUInt32LE(pos + 42);
    const name = buffer.subarray(pos + 46, pos + 46 + nameLength).toString("utf8");
    const dataStart = localOffset + 30 + buffer.readUInt16LE(localOffset + 26) + buffer.readUInt16LE(localOffset + 28);
    entries.push({ name, crc, data: buffer.subarray(dataStart, dataStart + size) });
    pos += 46 + nameLength;
  }
  return entries;
}

const server = await startServer();
const origin = `http://127.0.0.1:${server.address().port}`;
if (SYNTHETIC) {
  installSyntheticFaces();
  console.log("MODALITÀ SINTETICA: i volti vengono da un modello di prova, non da YuNet");
}
const golden = loadGoldenMask();
const originalPng = fs.readFileSync(path.join(fixtures, "astronaut.png")).toString("base64");
const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || undefined,
  args: process.env.CHROME_ARGS ? JSON.parse(process.env.CHROME_ARGS) : [],
  headless: true,
});
console.log(`Browser: ${browser.version()} - sito: ${origin}${BASE_PATH}`);

const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
const page = await openPage(context, origin);
let astronaut = null;

await check("la pagina si carica e i modelli vengono preparati senza errori", async () => {
  assert.equal(await page.getAttribute("body", "data-engine"), "ready", await page.textContent("#engine-status"));
  const text = await page.textContent("#capabilities");
  assert.match(text, /Soggetto e sfondo: attivo/);
  assert.match(text, /Volti: attivo/);
  assert.match(text, /Loghi: non attivo/);
  assert.equal(await page.isChecked("#background-toggle"), false, "l'opzione sfondo deve essere spenta di default");
  assert.deepEqual(page.problems, []);
  await page.screenshot({ path: path.join(resultsDir, "desktop-iniziale.png"), fullPage: true });
});

await check("sfondo intatto di default: volti e loghi oscurati, il resto dei pixel non cambia", async () => {
  astronaut = await upload(page, "astronaut.png");
  assert.notEqual(astronaut.state, "errore", JSON.stringify(astronaut.result.errors));
  const { result } = astronaut;
  assert.equal(result.width, 512);
  assert.equal(result.backgroundRedactedPx, 0, "con l'opzione spenta lo sfondo non deve essere toccato");
  if (SYNTHETIC) {
    assert.equal(astronaut.state, "da_rivedere", "il modello sintetico rileva sempre lo stesso volto, quindi la verifica finale deve segnalarlo");
    assert.equal(result.detections.volto, 1);
    assert.ok(result.detectionDetails.some((d) => d.redacted === false), "atteso un volto nel soggetto (visibile)");
    assert.ok(result.detectionDetails.some((d) => d.redacted === true), "atteso un volto fuori dal soggetto (oscurato)");
  } else {
    assert.equal(astronaut.state, "ok", JSON.stringify(result.errors));
  }
  const png = await downloadPng(page, astronaut.card);
  assert.deepEqual(scanPng(new Uint8Array(png)), [], "il PNG scaricato contiene metadati");
  assert.deepEqual(pngSize(png), { width: 512, height: 512 });
  const boxes = redactedBoxes(result.detectionDetails, 512, 512);
  const unchanged = await page.evaluate(async ({ b64, original, boxes, w, h }) => {
    const decode = async (bytes) => {
      const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);
      return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    };
    const toBytes = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    const out = await decode(toBytes(b64));
    const src = await decode(toBytes(original));
    const inBox = (x, y) => boxes.some(([x1, y1, x2, y2]) => x >= x1 && x < x2 && y >= y1 && y < y2);
    let same = 0;
    let total = 0;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (inBox(x, y)) continue;
        const i = (y * w + x) * 4;
        total += 1;
        if (out[i] === src[i] && out[i + 1] === src[i + 1] && out[i + 2] === src[i + 2]) same += 1;
      }
    }
    return same / total;
  }, { b64: png.toString("base64"), original: originalPng, boxes, w: 512, h: 512 });
  assert.ok(unchanged >= 0.9999, `pixel fuori dai riquadri non invariati: ${(unchanged * 100).toFixed(3)}%`);
  console.log(`      pixel fuori dai volti oscurati invariati: ${(unchanged * 100).toFixed(3)}%`);
});

await check("con l'opzione sfondo attiva la foto già caricata viene rielaborata e lo sfondo diventa nero", async () => {
  await page.check("#background-toggle");
  await waitDone(page, 0);
  const card = page.locator(".item").nth(0);
  const state = await card.getAttribute("data-state");
  const result = JSON.parse(await card.getAttribute("data-result"));
  assert.notEqual(state, "errore", JSON.stringify(result.errors));
  assert.ok(result.backgroundRedactedPx > 100000, `sfondo oscurato: ${result.backgroundRedactedPx} px`);
  const png = await downloadPng(page, card);
  assert.deepEqual(scanPng(new Uint8Array(png)), [], "il PNG scaricato contiene metadati");
  const stats = await page.evaluate(async ({ b64, original, goldenMask }) => {
    const decode = async (bytes) => {
      const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);
      return ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    };
    const toBytes = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    const out = await decode(toBytes(b64));
    const src = await decode(toBytes(original));
    let inter = 0;
    let union = 0;
    let same = 0;
    let compared = 0;
    for (let i = 0; i < goldenMask.length; i += 1) {
      if (src[i * 4] + src[i * 4 + 1] + src[i * 4 + 2] === 0) continue;
      const lit = out[i * 4] + out[i * 4 + 1] + out[i * 4 + 2] > 0;
      const ref = goldenMask[i] === "1";
      if (lit && ref) inter += 1;
      if (lit || ref) union += 1;
      if (lit) {
        compared += 1;
        if (out[i * 4] === src[i * 4] && out[i * 4 + 1] === src[i * 4 + 1] && out[i * 4 + 2] === src[i * 4 + 2]) same += 1;
      }
    }
    return { iou: inter / union, unchangedRatio: same / compared };
  }, { b64: png.toString("base64"), original: originalPng, goldenMask: golden.mask });
  assert.ok(stats.iou >= 0.98, `sovrapposizione con la maschera dell'algoritmo Python: ${stats.iou.toFixed(4)}`);
  assert.ok(stats.unchangedRatio >= 0.999, `pixel del soggetto non modificati: ${stats.unchangedRatio.toFixed(4)}`);
  console.log(`      IoU con la maschera Python ${stats.iou.toFixed(4)}, pixel del soggetto invariati ${(stats.unchangedRatio * 100).toFixed(2)}%`);
  await page.screenshot({ path: path.join(resultsDir, "desktop-risultato.png"), fullPage: true });
  await page.uncheck("#background-toggle");
  await waitDone(page, 0);
});

if (!SYNTHETIC && process.env.GOLDEN_FACES) {
  await check("i volti trovati nel browser coincidono con quelli di OpenCV (algoritmo originale)", async () => {
    const expected = JSON.parse(fs.readFileSync(process.env.GOLDEN_FACES, "utf8")).faces;
    const found = astronaut.result.detectionDetails.filter((d) => d.kind === "volto");
    assert.ok(expected.length > 0, "il riferimento OpenCV non contiene volti");
    const iou = (a, b) => {
      const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
      const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
      const inter = w > 0 && h > 0 ? w * h : 0;
      return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
    };
    for (const row of expected) {
      const box = [row[0], row[1], row[0] + row[2], row[1] + row[3]];
      const best = Math.max(0, ...found.map((d) => iou(box, d.box)));
      assert.ok(best >= 0.8, `volto OpenCV ${JSON.stringify(box.map(Math.round))} non trovato nel browser (IoU migliore ${best.toFixed(2)})`);
    }
    assert.equal(found.length, expected.length, `volti nel browser ${found.length}, in OpenCV ${expected.length}`);
  });
}

await check("decodifica di YuNet nel browser identica a quella di OpenCV (modelli sintetici, 9 casi)", async () => {
  const reference = JSON.parse(fs.readFileSync(path.join(fixtures, "yunet_decode", "reference.json"), "utf8"));
  const produced = await page.evaluate(async ({ cases, width, height, base, host }) => {
    const ort = await import(`${base}vendor/ort/ort.wasm.min.mjs`);
    ort.env.wasm.wasmPaths = `${base}vendor/ort/`;
    ort.env.wasm.numThreads = 1;
    const { decodeYuNet } = await import(`${base}js/faces.js`);
    const sessions = {};
    const out = [];
    for (const c of cases) {
      if (!sessions[c.model]) {
        const bytes = new Uint8Array(await (await fetch(`${host}/fixtures/yunet_decode/${c.model}`)).arrayBuffer());
        sessions[c.model] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      }
      const session = sessions[c.model];
      const results = await session.run({ input: new ort.Tensor("float32", new Float32Array(3 * height * width), [1, 3, height, width]) });
      const outputs = {};
      for (const name of session.outputNames) outputs[name] = results[name].data;
      const faces = decodeYuNet(outputs, { inputWidth: width, inputHeight: height, scoreThreshold: c.score_threshold, nmsThreshold: c.nms_threshold });
      out.push(faces.map((f) => [f.x, f.y, f.w, f.h, ...f.landmarks, f.score]));
    }
    return out;
  }, { cases: reference.cases, width: reference.width, height: reference.height, base: `${origin}${BASE_PATH}`, host: origin });
  let worst = 0;
  let compared = 0;
  reference.cases.forEach((c, i) => {
    assert.equal(produced[i].length, c.faces.length, `caso ${i}: ${produced[i].length} volti invece di ${c.faces.length}`);
    c.faces.forEach((row, k) => row.forEach((value, j) => {
      worst = Math.max(worst, Math.abs(value - produced[i][k][j]));
      compared += 1;
    }));
  });
  assert.ok(worst < 1e-3, `differenza massima ${worst}`);
  console.log(`      ${compared} valori confrontati con OpenCV, differenza massima ${worst.toExponential(2)}`);
});

await check("JPEG con EXIF, XMP e commento: i metadati vengono rilevati e non finiscono nel PNG", async () => {
  const { card, state, result } = await upload(page, "astronaut_exif.jpg");
  assert.notEqual(state, "errore", JSON.stringify(result.errors));
  for (const kind of ["exif", "xmp", "jpeg_comment"]) {
    assert.ok(result.metadataRemoved.includes(kind), `metadato non rilevato: ${kind} (${result.metadataRemoved})`);
  }
  assert.deepEqual(scanPng(new Uint8Array(await downloadPng(page, card))), []);
});

await check("l'orientamento EXIF viene applicato (40x20 con orientamento 6 diventa 20x40)", async () => {
  const { card, state, result } = await upload(page, "orientamento_6.jpg");
  assert.notEqual(state, "errore", JSON.stringify(result.errors));
  assert.deepEqual(pngSize(await downloadPng(page, card)), { width: 20, height: 40 });
  assert.ok(result.metadataRemoved.includes("exif"));
});

await check("un file che non è un'immagine viene rifiutato con un messaggio chiaro e si può riprovare", async () => {
  const { card, state, result } = await upload(page, "non_immagine.png");
  assert.equal(state, "errore");
  assert.match(result.errors.join(" "), /non è un'immagine/);
  assert.equal(await card.getByRole("button", { name: "Riprova" }).isVisible(), true);
  assert.equal(await card.getByRole("button", { name: "Scarica PNG" }).isDisabled(), true);
});

await check("lo ZIP con tutte le immagini contiene PNG puliti e un report valido", async () => {
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Scarica tutto (.zip)" }).click()]);
  const entries = readZip(fs.readFileSync(await download.path()));
  const names = entries.map((e) => e.name);
  assert.ok(names.includes("report.json"), names.join(", "));
  assert.ok(names.includes("astronaut_redacted.png") && names.includes("astronaut_exif_redacted.png"), names.join(", "));
  for (const entry of entries) {
    if (typeof zlib.crc32 === "function") assert.equal(zlib.crc32(entry.data), entry.crc, `CRC errato per ${entry.name}`);
    if (entry.name.endsWith(".png")) assert.deepEqual(scanPng(new Uint8Array(entry.data)), [], entry.name);
  }
  const report = JSON.parse(entries.find((e) => e.name === "report.json").data.toString("utf8"));
  assert.ok(report.files.length >= 3);
});

await check("il pulsante per caricare un'altra immagine e lo svuotamento dell'elenco funzionano", async () => {
  assert.equal(await page.locator('label[for="file-input"]', { hasText: "Carica un'altra immagine" }).isVisible(), true);
  await page.getByRole("button", { name: "Svuota elenco" }).click();
  assert.equal(await page.locator(".item").count(), 0);
});

await check("privacy: il browser non invia nulla, solo richieste GET allo stesso sito", async () => {
  const foreign = page.requests.filter((r) => !r.url.startsWith(origin) && !r.url.startsWith("blob:") && !r.url.startsWith("data:"));
  assert.deepEqual(foreign.map((r) => r.url), [], "richieste verso altri siti");
  const writes = page.requests.filter((r) => r.method !== "GET" || r.body);
  assert.deepEqual(writes.map((r) => `${r.method} ${r.url}`), [], "richieste con dati in uscita");
  const serverWrites = requestLog.filter((r) => r.method !== "GET");
  assert.deepEqual(serverWrites, [], "il server ha ricevuto richieste non GET");
  assert.deepEqual(page.problems, []);
});

await check("smartphone (390x844, touch): nessuno scorrimento orizzontale e risultato visibile", async () => {
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, acceptDownloads: true });
  const mobile = await openPage(phone, origin);
  assert.equal(await mobile.getAttribute("body", "data-engine"), "ready");
  const { state, result } = await upload(mobile, "astronaut.png");
  assert.notEqual(state, "errore", JSON.stringify(result.errors));
  const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert.ok(overflow <= 1, `scorrimento orizzontale di ${overflow}px`);
  await mobile.screenshot({ path: path.join(resultsDir, "smartphone-risultato.png"), fullPage: true });
  assert.deepEqual(mobile.problems, []);
  await phone.close();
});

await browser.close();
server.close();
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} prove superate${SYNTHETIC ? " (modalità sintetica)" : ""}`);
process.exit(failed.length ? 1 : 0);
