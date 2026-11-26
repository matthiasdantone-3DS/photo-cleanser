// Ridimensionamento bilineare per ingrandire; i bordi esterni contano come pixel neri se border = "zero"
export function resizeBilinear(src, sw, sh, dw, dh, border = "replicate") {
  const out = new Uint8ClampedArray(dw * dh * 4);
  const scaleX = sw / dw;
  const scaleY = sh / dh;
  for (let y = 0; y < dh; y += 1) {
    const fy = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.floor(fy);
    const wy = fy - y0;
    for (let x = 0; x < dw; x += 1) {
      const fx = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.floor(fx);
      const wx = fx - x0;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c += 1) {
        const v00 = sample(src, sw, sh, x0, y0, c, border);
        const v10 = sample(src, sw, sh, x0 + 1, y0, c, border);
        const v01 = sample(src, sw, sh, x0, y0 + 1, c, border);
        const v11 = sample(src, sw, sh, x0 + 1, y0 + 1, c, border);
        out[o + c] = (v00 * (1 - wx) + v10 * wx) * (1 - wy) + (v01 * (1 - wx) + v11 * wx) * wy;
      }
    }
  }
  return out;
}

// Legge un campione di un'immagine RGBA, replicando o azzerando i bordi
function sample(src, sw, sh, x, y, channel, border) {
  if (x < 0 || y < 0 || x >= sw || y >= sh) {
    if (border === "zero") {
      return 0;
    }
    x = Math.min(Math.max(x, 0), sw - 1);
    y = Math.min(Math.max(y, 0), sh - 1);
  }
  return src[(y * sw + x) * 4 + channel];
}

// Riduce un'immagine RGBA con la media delle aree (come cv2.INTER_AREA), una riga alla volta per risparmiare memoria
export function resizeArea(src, sw, sh, dw, dh) {
  const out = new Uint8ClampedArray(dw * dh * 4);
  const scaleX = sw / dw;
  const scaleY = sh / dh;
  const row = new Float32Array(dw * 4);
  const acc = new Float32Array(dw * 4);
  for (let y = 0; y < dh; y += 1) {
    const top = y * scaleY;
    const bottom = (y + 1) * scaleY;
    acc.fill(0);
    let weightSum = 0;
    for (let sy = Math.floor(top); sy < Math.min(sh, Math.ceil(bottom)); sy += 1) {
      const weightY = Math.min(bottom, sy + 1) - Math.max(top, sy);
      if (weightY <= 0) {
        continue;
      }
      weightSum += weightY;
      row.fill(0);
      for (let x = 0; x < dw; x += 1) {
        const left = x * scaleX;
        const right = (x + 1) * scaleX;
        let wsum = 0;
        for (let sx = Math.floor(left); sx < Math.min(sw, Math.ceil(right)); sx += 1) {
          const weightX = Math.min(right, sx + 1) - Math.max(left, sx);
          if (weightX <= 0) {
            continue;
          }
          const s = (sy * sw + sx) * 4;
          const o = x * 4;
          row[o] += src[s] * weightX;
          row[o + 1] += src[s + 1] * weightX;
          row[o + 2] += src[s + 2] * weightX;
          row[o + 3] += src[s + 3] * weightX;
          wsum += weightX;
        }
        const o = x * 4;
        for (let c = 0; c < 4; c += 1) {
          acc[o + c] += (row[o + c] / wsum) * weightY;
        }
      }
    }
    for (let i = 0; i < dw * 4; i += 1) {
      out[y * dw * 4 + i] = acc[i] / weightSum;
    }
  }
  return out;
}

// Ridimensiona scegliendo il metodo adatto: media delle aree se si riduce, bilineare se si ingrandisce
export function resizeRgba(src, sw, sh, dw, dh) {
  if (sw === dw && sh === dh) {
    return src;
  }
  return dw <= sw && dh <= sh ? resizeArea(src, sw, sh, dw, dh) : resizeBilinear(src, sw, sh, dw, dh);
}
