import { ACCEPTED_TYPES, LIMITS } from "./config.js";
import { EngineClient } from "./engine-client.js";
import { buildZip, uniqueName } from "./zip.js";

const STATE_LABELS = {
  queued: "In coda",
  processing: "In elaborazione",
  ok: "Oscurata",
  nessun_soggetto: "Nessun soggetto: non oscurata",
  da_rivedere: "Da rivedere",
  errore: "Errore",
};

const dom = {
  dropzone: document.getElementById("dropzone"),
  input: document.getElementById("file-input"),
  notice: document.getElementById("notice"),
  actions: document.getElementById("actions"),
  zipButton: document.getElementById("zip-button"),
  clearButton: document.getElementById("clear-button"),
  results: document.getElementById("results"),
  capabilities: document.getElementById("capabilities"),
  engineStatus: document.getElementById("engine-status"),
};

const items = [];
let engine = null;
let queueRunning = false;
let nextId = 1;

// Crea un elemento con classe e testo opzionali
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

// Mostra o nasconde l'avviso sui file scartati
function showNotice(messages) {
  if (!messages.length) {
    dom.notice.hidden = true;
    dom.notice.textContent = "";
    return;
  }
  dom.notice.textContent = `File non caricati: ${messages.join("; ")}.`;
  dom.notice.hidden = false;
}

// Disegna una funzione attiva o non attiva nella testata
function capabilityItem(label, on, hint) {
  const node = el("li", "", `${label}: ${on ? "attivo" : "non attivo"}`);
  node.dataset.on = on ? "true" : "false";
  if (hint) {
    node.title = hint;
  }
  return node;
}

// Aggiorna la testata con le fasi disponibili
function renderCapabilities(caps) {
  const reason = (phase) => {
    const found = caps.skipped.find((item) => item.phase === phase);
    return found ? found.reason : caps.segmenterError || "";
  };
  dom.capabilities.replaceChildren(
    capabilityItem("Volti", caps.faces, reason("volto")),
    capabilityItem("Soggetto e sfondo", caps.segmentation, caps.segmenterError || ""),
    capabilityItem("Loghi", caps.logos, reason("logo"))
  );
}

// Avvia il motore di Machine Learning in background mentre l'utente sceglie le foto
async function startEngine() {
  engine = new EngineClient((progress) => {
    dom.engineStatus.textContent = `${progress.message}…`;
  });
  try {
    const caps = await engine.start();
    renderCapabilities(caps);
    document.body.dataset.engine = "ready";
    dom.engineStatus.textContent = "Modelli pronti.";
    runQueue();
  } catch (error) {
    document.body.dataset.engine = "error";
    dom.engineStatus.textContent = `Impossibile preparare i modelli: ${error.message}. Ricarica la pagina o prova un altro browser.`;
    dom.capabilities.replaceChildren(el("li", "", "Modelli non disponibili"));
  }
}

// Costruisce la scheda di un file con anteprima dell'originale
function createItem(file) {
  const item = {
    id: nextId++,
    file,
    state: "queued",
    result: null,
    blob: null,
    resultUrl: null,
    originalUrl: URL.createObjectURL(file),
    nodes: {},
  };
  const root = el("article", "item");
  const head = el("div", "item-head");
  const chip = el("span", "chip");
  head.append(el("h2", "item-name", file.name), chip);

  const original = el("figure", "pane");
  original.append(el("figcaption", "", "Originale"));
  const originalImage = el("img");
  originalImage.alt = `Originale: ${file.name}`;
  originalImage.src = item.originalUrl;
  original.append(originalImage);

  const redacted = el("figure", "pane");
  redacted.append(el("figcaption", "", "Oscurata"));
  const resultBody = el("div", "pane-body");
  redacted.append(resultBody);

  const panes = el("div", "panes");
  panes.append(original, redacted);

  const note = el("p", "item-note");
  const soft = el("p", "item-note-soft");
  const issues = el("ul", "item-issues");

  const download = el("button", "button", "Scarica PNG");
  download.type = "button";
  download.disabled = true;
  download.addEventListener("click", () => downloadItem(item));
  const retry = el("button", "button button-quiet", "Riprova");
  retry.type = "button";
  retry.hidden = true;
  retry.addEventListener("click", () => retryItem(item));
  const remove = el("button", "button button-quiet", "Rimuovi");
  remove.type = "button";
  remove.addEventListener("click", () => removeItem(item));
  const actions = el("div", "item-actions");
  actions.append(download, retry, remove);

  root.append(head, panes, note, soft, issues, actions);
  item.nodes = { root, chip, resultBody, note, soft, issues, download, retry };
  return item;
}

// Riassume in una frase cosa è stato fatto sul file
function describeResult(item) {
  const result = item.result;
  if (!result) {
    return "";
  }
  const parts = [];
  if (item.state === "nessun_soggetto") {
    parts.push("Non è stato trovato nessun soggetto principale, quindi nulla è stato oscurato.");
  } else if (item.state !== "errore") {
    const counts = [];
    const faces = result.detections.volto || 0;
    const logos = result.detections.logo || 0;
    if (faces) {
      counts.push(`${faces} ${faces === 1 ? "volto" : "volti"}`);
    }
    if (logos) {
      counts.push(`${logos} ${logos === 1 ? "logo" : "loghi"}`);
    }
    parts.push(counts.length ? `Oscurati: ${counts.join(" e ")}. Sfondo oscurato.` : "Sfondo oscurato.");
  }
  if (item.blob) {
    const removed = result.metadataRemoved || [];
    parts.push(removed.length ? `Metadati rimossi: ${removed.join(", ")}.` : "Nessun metadato da rimuovere.");
  }
  return parts.join(" ");
}

// Aggiorna la scheda in base allo stato del file
function renderItem(item) {
  const { chip, resultBody, note, soft, issues, download, retry, root } = item.nodes;
  chip.textContent = STATE_LABELS[item.state] || item.state;
  chip.dataset.state = item.state;
  root.dataset.state = item.state;
  if (item.result) {
    root.dataset.result = JSON.stringify(item.result);
  }
  resultBody.replaceChildren();
  if (item.resultUrl) {
    const image = el("img");
    image.alt = `Oscurata: ${item.file.name}`;
    image.src = item.resultUrl;
    resultBody.append(image);
  } else if (item.state === "processing") {
    const waiting = el("p", "placeholder");
    waiting.append(el("span", "spinner"), document.createTextNode("Elaborazione in corso"));
    resultBody.append(waiting);
  } else if (item.state === "queued") {
    const text = document.body.dataset.engine === "loading" ? "In attesa dei modelli" : "In attesa";
    resultBody.append(el("p", "placeholder", text));
  } else {
    resultBody.append(el("p", "placeholder", "Nessuna immagine prodotta"));
  }
  note.textContent = describeResult(item);
  const result = item.result;
  soft.textContent = result && result.downscaled
    ? `Per i limiti del browser l'immagine è stata ridotta da ${result.originalWidth}x${result.originalHeight} a ${result.width}x${result.height} pixel.`
    : "";
  const messages = result && result.errors ? result.errors : [];
  issues.replaceChildren(...messages.map((message) => el("li", "", message)));
  download.disabled = !item.blob;
  retry.hidden = item.state !== "errore";
}

// Elabora un file nel motore e salva il risultato nella scheda
async function processItem(item) {
  item.state = "processing";
  renderItem(item);
  try {
    const { result, blob } = await engine.process(item.file);
    if (!items.includes(item)) {
      return;
    }
    if (blob) {
      item.blob = blob;
      item.resultUrl = URL.createObjectURL(blob);
    }
    item.result = result;
    item.state = result.status;
  } catch (error) {
    if (!items.includes(item)) {
      return;
    }
    item.result = { status: "errore", detections: {}, metadataRemoved: [], errors: [error.message || "Elaborazione non riuscita."] };
    item.state = "errore";
  }
  renderItem(item);
}

// Elabora le immagini una alla volta, appena i modelli sono pronti
async function runQueue() {
  if (queueRunning || !engine || document.body.dataset.engine !== "ready") {
    return;
  }
  queueRunning = true;
  try {
    let next = items.find((item) => item.state === "queued");
    while (next) {
      await processItem(next);
      updateActions();
      next = items.find((item) => item.state === "queued");
    }
  } finally {
    queueRunning = false;
    updateActions();
  }
}

// Mostra o nasconde i pulsanti globali in base all'elenco
function updateActions() {
  dom.actions.hidden = items.length === 0;
  dom.zipButton.disabled = items.filter((item) => item.blob).length < 1;
}

// Aggiunge i file scelti, scartando quelli non adatti
function addFiles(fileList) {
  const accepted = [];
  const skipped = [];
  for (const file of Array.from(fileList)) {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      skipped.push(`${file.name} (formato non supportato)`);
    } else if (file.size > LIMITS.maxFileBytes) {
      skipped.push(`${file.name} (supera ${Math.round(LIMITS.maxFileBytes / 1048576)} MB)`);
    } else {
      accepted.push(file);
    }
  }
  const room = Math.max(0, LIMITS.maxFiles - items.length);
  if (accepted.length > room) {
    skipped.push(`${accepted.length - room} oltre il limite di ${LIMITS.maxFiles} immagini`);
    accepted.length = room;
  }
  for (const file of accepted) {
    const item = createItem(file);
    items.push(item);
    dom.results.append(item.nodes.root);
    renderItem(item);
  }
  showNotice(skipped);
  updateActions();
  runQueue();
}

// Salva un Blob come file scaricato
function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Nome del file di output: <originale>_redacted.png
function outputName(item) {
  const dot = item.file.name.lastIndexOf(".");
  const stem = dot > 0 ? item.file.name.slice(0, dot) : item.file.name;
  return `${stem || "immagine"}_redacted.png`;
}

// Scarica il PNG oscurato di una scheda
function downloadItem(item) {
  if (item.blob) {
    saveBlob(item.blob, outputName(item));
  }
}

// Rimette in coda un file andato in errore
function retryItem(item) {
  item.state = "queued";
  item.result = null;
  item.blob = null;
  if (item.resultUrl) {
    URL.revokeObjectURL(item.resultUrl);
    item.resultUrl = null;
  }
  renderItem(item);
  runQueue();
}

// Toglie una scheda dall'elenco e libera la memoria
function removeItem(item) {
  const index = items.indexOf(item);
  if (index === -1) {
    return;
  }
  items.splice(index, 1);
  URL.revokeObjectURL(item.originalUrl);
  if (item.resultUrl) {
    URL.revokeObjectURL(item.resultUrl);
  }
  item.blob = null;
  item.nodes.root.remove();
  updateActions();
}

// Svuota tutto l'elenco
function clearAll() {
  for (const item of [...items]) {
    removeItem(item);
  }
  showNotice([]);
}

// Scarica tutte le immagini oscurate in un unico ZIP, con un report JSON
async function downloadZip() {
  dom.zipButton.disabled = true;
  try {
    const used = new Set(["report.json"]);
    const entries = [];
    const report = [];
    for (const item of items) {
      if (!item.result) {
        continue;
      }
      let name = null;
      if (item.blob) {
        name = uniqueName(outputName(item), used);
        entries.push({ name, data: new Uint8Array(await item.blob.arrayBuffer()) });
      }
      const { detectionDetails, ...summary } = item.result;
      report.push({ ...summary, input: item.file.name, output: name, detectionDetails });
    }
    entries.push({ name: "report.json", data: new TextEncoder().encode(JSON.stringify({ files: report }, null, 2)) });
    saveBlob(buildZip(entries), "redactor.zip");
  } finally {
    updateActions();
  }
}

["dragenter", "dragover"].forEach((type) => {
  dom.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dom.dropzone.dataset.active = "true";
  });
});

["dragleave", "drop"].forEach((type) => {
  dom.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dom.dropzone.dataset.active = "false";
  });
});

dom.dropzone.addEventListener("drop", (event) => {
  if (event.dataTransfer && event.dataTransfer.files.length) {
    addFiles(event.dataTransfer.files);
  }
});

// Evita che il browser apra un file lasciato cadere fuori dalla zona di caricamento
["dragover", "drop"].forEach((type) => {
  window.addEventListener(type, (event) => event.preventDefault());
});

dom.input.addEventListener("change", () => {
  addFiles(dom.input.files);
  dom.input.value = "";
});
dom.zipButton.addEventListener("click", downloadZip);
dom.clearButton.addEventListener("click", clearAll);

startEngine();
