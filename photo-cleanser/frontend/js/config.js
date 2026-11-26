// Soglie dell'algoritmo: gli stessi valori di config.py (Settings) nella versione Python
export const SETTINGS = {
  faceScore: 0.6,
  logoConf: 0.4,
  maskThreshold: 0.5,
  maskMinComponentRatio: 0.02,
  padPx: 6,
  morphKernel: 5,
};

// Limiti pensati per i browser, in particolare per gli smartphone
export const LIMITS = {
  maxFileBytes: 25 * 1024 * 1024,
  maxFiles: 50,
  maxSide: 4096,
  maxPixels: 16777216,
  maxDecodePixels: 120000000,
};

// Formati accettati
export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
