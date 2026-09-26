function drawLineChart(canvas, series, options = {}) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 300;
  const cssHeight = canvas.clientHeight || 160;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padding = { top: 10, right: 10, bottom: 20, left: 40 };
  const plotW = cssWidth - padding.left - padding.right;
  const plotH = cssHeight - padding.top - padding.bottom;

  const color = options.color || "#4f8cff";
  const values = series.filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (values.length < 2) {
    ctx.fillStyle = "#8b8d97";
    ctx.font = "12px sans-serif";
    ctx.fillText("Waiting for data...", padding.left, cssHeight / 2);
    return;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const rangePad = (max - min) * 0.1;
  min -= rangePad;
  max += rangePad;

  // Gridlines
  ctx.strokeStyle = "#33363f";
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, Consolas, monospace";
  ctx.fillStyle = "#8b8d97";
  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y = padding.top + (plotH * i) / gridLines;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(cssWidth - padding.right, y);
    ctx.stroke();
    const val = max - ((max - min) * i) / gridLines;
    ctx.fillText(val.toFixed(1), 2, y + 3);
  }

  // Line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  const n = series.length;
  series.forEach((v, i) => {
    if (typeof v !== "number" || Number.isNaN(v)) return;
    const x = padding.left + (plotW * i) / Math.max(1, n - 1);
    const y = padding.top + plotH - ((v - min) / (max - min)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under curve
  ctx.lineTo(padding.left + plotW, padding.top + plotH);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.closePath();
  ctx.fillStyle = color + "22";
  ctx.fill();
}
