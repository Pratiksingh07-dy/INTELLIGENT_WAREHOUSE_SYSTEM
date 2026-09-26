
const COLOR = {
  arm: "#f2b705",
  robot: "#7fa8c9",
  drone: "#a996d1",
  queued: "#c7c4b8",
  completed: "#57a373",
  highPriority: "#d6564a",
  busyBadge: "#f2b705",
  idleBadge: "#5c5e68",
  chargeBadge: "#d6564a",
  line: "#33363f",
  ink: "#e9e6dc",
  inkDim: "#8b8d97",
};

function renderWarehouse(canvas, state) {
  if (!canvas || !state) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = canvas.clientHeight || 440;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.textBaseline = "alphabetic";

  const zoneW = cssWidth / 3;
  const skyH = 56;                 // dedicated lane for drones - never shared with labels
  const zoneY = skyH + 8;
  const trackH = 60;               // dedicated lane for ground robots
  const zoneH = cssHeight - zoneY - trackH - 14;

  const zones = [
    { key: "storage", label: "Storage", x: 0 },
    { key: "packing", label: "Packing", x: zoneW },
    { key: "dispatch", label: "Dispatch", x: zoneW * 2 },
  ];
  const zoneCenterX = { storage: zoneW / 2, packing: zoneW * 1.5, dispatch: zoneW * 2.5 };

  // --- Zone floor panels ---
  zones.forEach((z) => {
    ctx.fillStyle = "#171820";
    ctx.strokeStyle = COLOR.line;
    ctx.lineWidth = 1;
    ctx.fillRect(z.x + 6, zoneY, zoneW - 12, zoneH);
    ctx.strokeRect(z.x + 6.5, zoneY + 0.5, zoneW - 13, zoneH - 1);

    ctx.fillStyle = COLOR.inkDim;
    ctx.font = "600 11px 'Trebuchet MS', Arial, sans-serif";
    ctx.fillText(z.label, z.x + 16, zoneY + 20);
  });

  // Connecting rail along the floor between zones
  ctx.strokeStyle = COLOR.line;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(6, zoneY + zoneH + trackH / 2);
  ctx.lineTo(cssWidth - 6, zoneY + zoneH + trackH / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // --- Storage: queue ---
  const queue = state.queue || [];
  ctx.fillStyle = COLOR.ink;
  ctx.font = "12px 'Trebuchet MS', Arial, sans-serif";
  ctx.fillText(`Queue  ${queue.length}`, 16, zoneY + 40);
  queue.slice(0, 12).forEach((order, i) => {
    const col = i % 6;
    const row = Math.floor(i / 6);
    const bx = 16 + col * 20;
    const by = zoneY + 52 + row * 20;
    ctx.fillStyle = order.priority === "high" ? COLOR.highPriority : COLOR.queued;
    ctx.fillRect(bx, by, 14, 14);
  });

  // --- Packing: robotic arms, centered with breathing room ---
  const arms = state.arms || [];
  const armY = zoneY + zoneH / 2 + 6;
  const armSpacing = 76;
  const armStartX = zoneCenterX.packing - ((arms.length - 1) * armSpacing) / 2;
  arms.forEach((arm, i) => {
    const ax = armStartX + i * armSpacing;
    drawUnit(ctx, ax, armY, COLOR.arm, `A${arm.arm_id}`, arm.status === "processing");
    ctx.fillStyle = COLOR.inkDim;
    ctx.font = "10px 'Trebuchet MS', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(arm.status, ax, armY + 34);
    ctx.textAlign = "left";
  });

  // --- Dispatch: completed orders ---
  const completed = state.recent_completed || [];
  ctx.fillStyle = COLOR.ink;
  ctx.font = "12px 'Trebuchet MS', Arial, sans-serif";
  const dispatchedTotal = state.metrics ? state.metrics.total_orders_completed : 0;
  ctx.fillText(`Dispatched  ${dispatchedTotal}`, zoneW * 2 + 16, zoneY + 40);
  completed.slice(-12).forEach((order, i) => {
    const col = i % 6;
    const row = Math.floor(i / 6);
    const bx = zoneW * 2 + 16 + col * 20;
    const by = zoneY + 52 + row * 20;
    ctx.fillStyle = COLOR.completed;
    ctx.fillRect(bx, by, 14, 14);
  });

  // --- Sky lane: drones, spread out with a tether line to their zone ---
  const drones = state.drones || [];
  const droneSpacingPerZone = 34;
  const droneCountByZone = {};
  drones.forEach((drone) => {
    const zoneKey = drone.current_zone || "storage";
    const idx = droneCountByZone[zoneKey] || 0;
    droneCountByZone[zoneKey] = idx + 1;
    const baseX = zoneCenterX[zoneKey] ?? zoneCenterX.storage;
    const dx = baseX + (idx - 0.5) * droneSpacingPerZone;
    const dy = 26;
    const active = drone.status === "scanning" || drone.status === "monitoring";

    // tether down to the zone it's scanning
    ctx.strokeStyle = COLOR.line;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(dx, dy + 10);
    ctx.lineTo(dx, zoneY);
    ctx.stroke();
    ctx.setLineDash([]);

    drawDrone(ctx, dx, dy, COLOR.drone, `D${drone.drone_id}`, active);
  });

  // --- Track lane: ground robots ---
  const robots = state.robots || [];
  const trackY = zoneY + zoneH + trackH / 2;
  const robotCountByZone = {};
  robots.forEach((robot) => {
    const zoneKey = robot.location || "storage";
    const idx = robotCountByZone[zoneKey] || 0;
    robotCountByZone[zoneKey] = idx + 1;
    const baseX = zoneCenterX[zoneKey] ?? zoneCenterX.storage;
    const rx = baseX + (idx - 0.5) * 40;
    const moving = robot.status === "moving";
    drawRobot(ctx, rx, trackY, COLOR.robot, `G${robot.robot_id}`, robot.status !== "idle", moving);
  });

  drawLegend(ctx, cssWidth, cssHeight);
}

/** Circular unit (used for robotic arms) with a small busy/idle badge. */
function drawUnit(ctx, x, y, color, label, busy) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 17, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#14151a";
  ctx.font = "700 11px Consolas, 'SFMono-Regular', monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + 4);
  ctx.textAlign = "left";
  drawBadge(ctx, x + 12, y - 12, busy);
}

/** Rounded unit (used for ground robots) with a directional chevron when moving. */
function drawRobot(ctx, x, y, color, label, active, moving) {
  ctx.fillStyle = color;
  roundRect(ctx, x - 16, y - 11, 32, 22, 4);
  ctx.fill();
  ctx.fillStyle = "#14151a";
  ctx.font = "700 10px Consolas, 'SFMono-Regular', monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + 4);
  ctx.textAlign = "left";
  if (moving) {
    ctx.fillStyle = "#14151a";
    ctx.beginPath();
    ctx.moveTo(x + 18, y - 4);
    ctx.lineTo(x + 24, y);
    ctx.lineTo(x + 18, y + 4);
    ctx.closePath();
    ctx.fill();
  }
  drawBadge(ctx, x + 13, y - 13, active);
}

/** Triangular unit (used for drones). */
function drawDrone(ctx, x, y, color, label, active) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x, y - 9);
  ctx.lineTo(x + 9, y + 7);
  ctx.lineTo(x - 9, y + 7);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = COLOR.ink;
  ctx.font = "9px Consolas, 'SFMono-Regular', monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, y + 20);
  ctx.textAlign = "left";
  drawBadge(ctx, x + 8, y - 8, active);
}

/** Small status badge: amber ring = busy, dim grey dot = idle/available. */
function drawBadge(ctx, x, y, busy) {
  ctx.fillStyle = busy ? COLOR.busyBadge : COLOR.idleBadge;
  ctx.beginPath();
  ctx.arc(x, y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#14151a";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawLegend(ctx, width, height) {
  const items = [
    { color: COLOR.arm, label: "Arm" },
    { color: COLOR.robot, label: "Ground robot" },
    { color: COLOR.drone, label: "Drone" },
    { color: COLOR.completed, label: "Completed" },
    { color: COLOR.highPriority, label: "High priority" },
  ];
  let x = 16;
  const y = height - 8;
  ctx.font = "10px 'Trebuchet MS', Arial, sans-serif";
  items.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y - 8, 8, 8);
    ctx.fillStyle = COLOR.inkDim;
    ctx.fillText(item.label, x + 12, y);
    x += ctx.measureText(item.label).width + 28;
  });
}
