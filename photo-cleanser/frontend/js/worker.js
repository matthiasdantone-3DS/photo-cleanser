import { createEngine } from "./models.js";

let engine = null;

// Il lavoro pesante gira in un worker: la pagina resta fluida mentre il modello elabora
self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "init") {
    try {
      engine = await createEngine((progress) => self.postMessage({ type: "progress", ...progress }), new URL("../", self.location.href));
      self.postMessage({ type: "ready", capabilities: engine.capabilities });
    } catch (error) {
      self.postMessage({ type: "fatal", message: error.message });
    }
  } else if (message.type === "process") {
    try {
      const { result, blob } = await engine.pipeline.process(message.file);
      self.postMessage({ type: "done", id: message.id, result, blob });
    } catch (error) {
      self.postMessage({ type: "done", id: message.id, result: { status: "errore", errors: [error.message], detections: {}, metadataRemoved: [] }, blob: null });
    }
  }
};
