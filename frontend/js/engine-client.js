// Collega la pagina al motore: usa un worker e, se il browser non lo permette, lavora nella pagina stessa
export class EngineClient {
  constructor(onProgress) {
    this.onProgress = onProgress;
    this.pending = new Map();
    this.nextId = 1;
    this.worker = null;
    this.local = null;
    this.capabilities = null;
  }

  // Avvia il motore e restituisce le funzioni disponibili
  async start() {
    try {
      this.capabilities = await this.startWorker();
    } catch (error) {
      if (this.worker) {
        this.worker.terminate();
        this.worker = null;
      }
      const { createEngine } = await import("./models.js");
      this.local = await createEngine(this.onProgress, new URL("../", import.meta.url));
      this.capabilities = this.local.capabilities;
    }
    return this.capabilities;
  }

  // Avvia il worker e aspetta il messaggio "ready"
  startWorker() {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./worker.bundle.js", import.meta.url));
      this.worker = worker;
      worker.onerror = (event) => reject(new Error(event.message || "worker non avviabile"));
      worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "progress") {
          this.onProgress(message);
        } else if (message.type === "ready") {
          worker.onerror = null;
          worker.onmessage = (inner) => this.handle(inner.data);
          resolve(message.capabilities);
        } else if (message.type === "fatal") {
          reject(new Error(message.message));
        }
      };
      worker.postMessage({ type: "init" });
    });
  }

  // Smista le risposte del worker alle richieste in attesa
  handle(message) {
    if (message.type === "done" && this.pending.has(message.id)) {
      const resolve = this.pending.get(message.id);
      this.pending.delete(message.id);
      resolve({ result: message.result, blob: message.blob });
    }
  }

  // Elabora un file e restituisce { result, blob }
  process(file) {
    if (this.local) {
      return this.local.pipeline.process(file);
    }
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      this.worker.postMessage({ type: "process", id, file });
    });
  }
}
