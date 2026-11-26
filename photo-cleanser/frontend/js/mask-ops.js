// Filtro a finestra rettangolare separabile su una maschera 0/1: "dilate" o "erode", con i bordi ignorati come in OpenCV
function windowFilter(src, w, h, radius, mode) {
  const horizontal = new Uint8Array(w * h);
  for (let y = 0; y < h; y += 1) {
    const row = y * w;
    let count = 0;
    for (let x = 0; x <= Math.min(radius, w - 1); x += 1) {
      count += src[row + x];
    }
    for (let x = 0; x < w; x += 1) {
      const size = Math.min(x + radius, w - 1) - Math.max(x - radius, 0) + 1;
      horizontal[row + x] = mode === "dilate" ? (count > 0 ? 1 : 0) : count === size ? 1 : 0;
      if (x + radius + 1 < w) {
        count += src[row + x + radius + 1];
      }
      if (x - radius >= 0) {
        count -= src[row + x - radius];
      }
    }
  }
  const out = new Uint8Array(w * h);
  const counts = new Int32Array(w);
  for (let y = 0; y <= Math.min(radius, h - 1); y += 1) {
    for (let x = 0; x < w; x += 1) {
      counts[x] += horizontal[y * w + x];
    }
  }
  for (let y = 0; y < h; y += 1) {
    const size = Math.min(y + radius, h - 1) - Math.max(y - radius, 0) + 1;
    for (let x = 0; x < w; x += 1) {
      out[y * w + x] = mode === "dilate" ? (counts[x] > 0 ? 1 : 0) : counts[x] === size ? 1 : 0;
    }
    if (y + radius + 1 < h) {
      for (let x = 0; x < w; x += 1) {
        counts[x] += horizontal[(y + radius + 1) * w + x];
      }
    }
    if (y - radius >= 0) {
      for (let x = 0; x < w; x += 1) {
        counts[x] -= horizontal[(y - radius) * w + x];
      }
    }
  }
  return out;
}

// Chiusura morfologica con kernel quadrato (dilatazione e poi erosione), come cv2.MORPH_CLOSE con np.ones((k, k))
export function closeSquare(mask, w, h, kernel) {
  const radius = Math.floor(kernel / 2);
  return windowFilter(windowFilter(mask, w, h, radius, "dilate"), w, h, radius, "erode");
}

// Trova la radice di un'etichetta con compressione dei cammini
function findRoot(parent, label) {
  let root = label;
  while (parent[root] !== root) {
    root = parent[root];
  }
  while (parent[label] !== root) {
    const next = parent[label];
    parent[label] = root;
    label = next;
  }
  return root;
}

// Tiene solo le componenti connesse (8 vicini) con area almeno minRatio volte quella della più grande
export function keepLargeComponents(mask, w, h, minRatio) {
  const labels = new Int32Array(w * h);
  const parent = new Int32Array(Math.ceil(w / 2) * Math.ceil(h / 2) + 2);
  let next = 1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      if (!mask[i]) {
        continue;
      }
      const neighbours = [];
      if (x > 0 && labels[i - 1]) neighbours.push(labels[i - 1]);
      if (y > 0) {
        if (x > 0 && labels[i - w - 1]) neighbours.push(labels[i - w - 1]);
        if (labels[i - w]) neighbours.push(labels[i - w]);
        if (x < w - 1 && labels[i - w + 1]) neighbours.push(labels[i - w + 1]);
      }
      if (neighbours.length === 0) {
        parent[next] = next;
        labels[i] = next;
        next += 1;
      } else {
        let smallest = findRoot(parent, neighbours[0]);
        for (let n = 1; n < neighbours.length; n += 1) {
          const root = findRoot(parent, neighbours[n]);
          if (root < smallest) smallest = root;
        }
        labels[i] = smallest;
        for (const neighbour of neighbours) {
          parent[findRoot(parent, neighbour)] = smallest;
        }
      }
    }
  }
  if (next === 1) {
    return mask;
  }
  const areas = new Float64Array(next);
  for (let i = 0; i < labels.length; i += 1) {
    if (labels[i]) {
      const root = findRoot(parent, labels[i]);
      labels[i] = root;
      areas[root] += 1;
    }
  }
  let largest = 0;
  for (let label = 1; label < next; label += 1) {
    largest = Math.max(largest, areas[label]);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < labels.length; i += 1) {
    if (labels[i] && areas[labels[i]] >= largest * minRatio) {
      out[i] = 1;
    }
  }
  return out;
}

// Porta la maschera 256x256 alla dimensione dell'immagine (bilineare) e la binarizza con la soglia, senza array in virgola mobile grandi
export function thresholdUpsample(prob, pw, ph, w, h, threshold) {
  const out = new Uint8Array(w * h);
  const x0s = new Int32Array(w);
  const x1s = new Int32Array(w);
  const wxs = new Float32Array(w);
  for (let x = 0; x < w; x += 1) {
    const fx = (x + 0.5) * (pw / w) - 0.5;
    const x0 = Math.floor(fx);
    wxs[x] = fx - x0;
    x0s[x] = Math.min(Math.max(x0, 0), pw - 1);
    x1s[x] = Math.min(Math.max(x0 + 1, 0), pw - 1);
  }
  for (let y = 0; y < h; y += 1) {
    const fy = (y + 0.5) * (ph / h) - 0.5;
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    const r0 = Math.min(Math.max(y0, 0), ph - 1) * pw;
    const r1 = Math.min(Math.max(y0 + 1, 0), ph - 1) * pw;
    for (let x = 0; x < w; x += 1) {
      const wx = wxs[x];
      const top = prob[r0 + x0s[x]] * (1 - wx) + prob[r0 + x1s[x]] * wx;
      const bottom = prob[r1 + x0s[x]] * (1 - wx) + prob[r1 + x1s[x]] * wx;
      out[y * w + x] = top * (1 - wy) + bottom * wy >= threshold ? 1 : 0;
    }
  }
  return out;
}

// Costruisce la maschera del soggetto: soglia, chiusura dei buchi piccoli e filtro delle componenti piccole
export function buildSubjectMask(binary, w, h, { morphKernel = 5, minComponentRatio = 0.02 } = {}) {
  let any = false;
  for (let i = 0; i < binary.length; i += 1) {
    if (binary[i]) {
      any = true;
      break;
    }
  }
  if (!any) {
    return { mask: binary, empty: true };
  }
  const closed = closeSquare(binary, w, h, morphKernel);
  return { mask: keepLargeComponents(closed, w, h, minComponentRatio), empty: false };
}

// Dice se il centro di un riquadro cade dentro la maschera del soggetto
export function centerInsideMask(box, mask, w, h) {
  const cx = Math.min(Math.max(Math.floor((box[0] + box[2]) / 2), 0), w - 1);
  const cy = Math.min(Math.max(Math.floor((box[1] + box[3]) / 2), 0), h - 1);
  return mask[cy * w + cx] === 1;
}
