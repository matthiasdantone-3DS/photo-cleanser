// Soppressione dei non-massimi su riquadri [x1, y1, x2, y2], con la stessa logica di cv::dnn::NMSBoxes
export function nms(boxes, scores, { scoreThreshold, iouThreshold, topK = 0 }) {
  const order = [];
  for (let i = 0; i < scores.length; i += 1) {
    if (scores[i] > scoreThreshold) {
      order.push(i);
    }
  }
  order.sort((a, b) => scores[b] - scores[a]);
  const candidates = topK > 0 ? order.slice(0, topK) : order;
  const kept = [];
  for (const index of candidates) {
    let keep = true;
    for (const other of kept) {
      if (iou(boxes[index], boxes[other]) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      kept.push(index);
    }
  }
  return kept;
}

// Rapporto tra area di intersezione e area di unione di due riquadri
export function iou(a, b) {
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  if (areaA + areaB <= Number.EPSILON) {
    return 0;
  }
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const inter = width > 0 && height > 0 ? width * height : 0;
  return inter / (areaA + areaB - inter);
}
