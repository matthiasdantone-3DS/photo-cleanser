// Espande un riquadro di pad pixel senza uscire dai bordi
export function expandBox(box, pad, width, height) {
  return [Math.max(0, box[0] - pad), Math.max(0, box[1] - pad), Math.min(width, box[2] + pad), Math.min(height, box[3] + pad)];
}

// Annerisce sul posto sfondo (fuori dalla maschera) e riquadri; restituisce il numero di pixel di sfondo oscurati
export function redactInPlace(rgba, width, height, mask, boxes, pad) {
  let backgroundPx = 0;
  if (mask) {
    for (let i = 0; i < mask.length; i += 1) {
      if (mask[i] === 0) {
        const o = i * 4;
        rgba[o] = 0;
        rgba[o + 1] = 0;
        rgba[o + 2] = 0;
        backgroundPx += 1;
      }
    }
  }
  for (const box of boxes) {
    const [x1, y1, x2, y2] = expandBox(box, pad, width, height);
    for (let y = y1; y < y2; y += 1) {
      for (let x = x1; x < x2; x += 1) {
        const o = (y * width + x) * 4;
        rgba[o] = 0;
        rgba[o + 1] = 0;
        rgba[o + 2] = 0;
      }
    }
  }
  return backgroundPx;
}
