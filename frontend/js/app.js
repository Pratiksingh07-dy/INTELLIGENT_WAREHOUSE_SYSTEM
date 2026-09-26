const API = "/api";
let pollTimer = null;
let latestState = null;

// handle dashboard tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "compare") loadCompareTable();
    if (btn.dataset.view === "training") loadHistory();
  });
});

// send POST request to backend
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

// send GET request to backend
async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

// format numbers for display
function fmt(n, decimals = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return Number(n).toFixed(decimals);
}

// handle warehouse controls
document.getElementById("btn-start").addEventListener("click", () => postJSON(`${API}/control/start`));
document.getElementById("btn-pause").addEventListener("click", () => postJSON(`${API}/control/pause`));
document.getElementById("btn-reset").addEventListener("click", () => postJSON(`${API}/control/reset`));
document.getElementById("btn-generate").addEventListener("click", () => postJSON(`${API}/control/generate-orders`, { count: 5 }));

const speedSlider = document.getElementById("speed-slider");
speedSlider.addEventListener("input", () => {
  const speed = parseFloat(speedSlider.value);
  document.getElementById("speed-label").innerText = `${speed.toFixed(1)}x`;
  postJSON(`${API}/control/speed`, { speed });
});

// fetch live warehouse state
async function pollState() {
  try {
    const state = await getJSON(`${API}/state`);
    latestState = state;
    renderDashboard(state);
  } catch (e) {
    console.error("poll failed", e);
  }
}

// update dashboard with current state
function renderDashboard(state) {
  const m = state.metrics;
  document.getElementById("sim-time-badge").innerText = `t = ${fmt(state.time, 1)}`;
  document.getElementById("active-algo-label").innerText = `policy: ${state.active_algorithm}`;
  document.getElementById("running-label").innerText = state.running ? "Running" : "Stopped";
  document.getElementById("running-dot").classList.toggle("live", !!state.running);

  const stats = [
    { label: "Total orders", value: m.total_orders_generated },
    { label: "Pending", value: m.pending_orders },
    { label: "Completed", value: m.total_orders_completed },
    { label: "Throughput", value: fmt(m.throughput, 3) },
    { label: "Avg waiting time", value: fmt(m.average_waiting_time, 2) },
    { label: "Utilization", value: `${fmt(m.resource_utilization * 100, 1)}%` },
    { label: "Energy used", value: fmt(m.energy_used, 1) },
    { label: "Reward", value: fmt(m.last_reward, 2), cls: m.last_reward >= 0 ? "positive" : "negative" },
  ];
  const grid = document.getElementById("stats-grid");
  grid.innerHTML = stats.map(
    (s) => `<div class="gauge"><div class="g-label">${s.label}</div><div class="g-value ${s.cls || ""}">${s.value}</div></div>`
  ).join("");

  renderResourceList("arms-list", state.arms, (a) => ({
    id: `A${a.arm_id}`,
    status: a.status,
    active: a.status === "processing",
    stats: [`${(a.utilization * 100).toFixed(0)}% util`, `${a.tasks_completed} done`],
  }));
  renderResourceList("robots-list", state.robots, (r) => ({
    id: `G${r.robot_id}`,
    status: `${r.status} (${r.location}${r.destination ? " to " + r.destination : ""})`,
    active: r.status !== "idle",
    battery: r.battery,
  }));
  renderResourceList("drones-list", state.drones, (d) => ({
    id: `D${d.drone_id}`,
    status: `${d.status} - ${d.current_zone}`,
    active: d.status === "scanning" || d.status === "monitoring",
    battery: d.battery,
  }));

  renderOrderTable(state.queue.concat(state.transport_queue));

  renderWarehouse(document.getElementById("warehouse-canvas"), state);

  const h = state.history;
  drawLineChart(document.getElementById("chart-reward"), h.reward, { color: "#f2b705" });
  drawLineChart(document.getElementById("chart-queue"), h.queue_length, { color: "#d6564a" });
  drawLineChart(document.getElementById("chart-wait"), h.waiting_time, { color: "#a996d1" });
  drawLineChart(document.getElementById("chart-throughput"), h.throughput, { color: "#57a373" });
  drawLineChart(document.getElementById("chart-energy"), h.energy, { color: "#7fa8c9" });
  drawLineChart(document.getElementById("chart-utilization"), h.utilization, { color: "#f2b705" });
}

// render warehouse resources
function renderResourceList(elementId, items, mapper) {
  const el = document.getElementById(elementId);
  el.innerHTML = (items || []).map((item) => {
    const info = mapper(item);
    const rightCell = info.battery !== undefined
      ? `<div class="battery-track"><div class="battery-fill" style="width:${info.battery}%"></div></div>`
      : `<div class="r-stat">${(info.stats || []).join(" / ")}</div>`;
    return `<div class="resource-row">
      <span class="r-id">${info.id}</span>
      <span class="r-status${info.active ? " active" : ""}">${info.status}</span>
      ${rightCell}
    </div>`;
  }).join("") || `<div style="color:var(--ink-dim); font-size:13px;">No resources</div>`;
}

// render current order queue
function renderOrderTable(orders) {
  const tbody = document.querySelector("#order-table tbody");
  if (!orders || orders.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--ink-dim)">Queue is empty</td></tr>`;
    return;
  }

  tbody.innerHTML = orders.slice(0, 15).map((o) => `
    <tr>
      <td>#${o.order_id}</td>
      <td>${o.product}</td>
      <td class="priority-${o.priority}">${o.priority}</td>
      <td>${o.stage}</td>
      <td>${fmt(o.waiting_time, 1)}</td>
    </tr>
  `).join("");
}

// start RL training
document.getElementById("btn-train").addEventListener("click", async () => {
  const payload = {
    algorithm: document.getElementById("algo-select").value,
    num_episodes: parseInt(document.getElementById("param-episodes").value, 10),
    alpha: parseFloat(document.getElementById("param-alpha").value),
    gamma: parseFloat(document.getElementById("param-gamma").value),
    epsilon_start: parseFloat(document.getElementById("param-epsilon").value),
    epsilon_decay: parseFloat(document.getElementById("param-epsilon-decay").value),
  };
  const res = await postJSON(`${API}/train`, payload);
  if (res.detail) {
    document.getElementById("train-message").innerText = res.detail;
    return;
  }

  pollTrainingStatus();
});

let trainingPollTimer = null;

// check training progress
function pollTrainingStatus() {
  clearInterval(trainingPollTimer);
  trainingPollTimer = setInterval(async () => {
    const status = await getJSON(`${API}/train/status`);
    document.getElementById("train-message").innerText = status.message;
    document.getElementById("train-progress").style.width = `${(status.progress * 100).toFixed(0)}%`;
    if (!status.running) {
      clearInterval(trainingPollTimer);
      loadHistory();
      if (status.last_run_id) loadRunCurve(status.last_run_id);
    }
  }, 700);
}

// load previous training runs
async function loadHistory() {
  const data = await getJSON(`${API}/train/history?limit=20`);
  const tbody = document.querySelector("#history-table tbody");
  if (!data.runs || data.runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="no-results">No training runs yet</td></tr>`;
    return;
  }

  tbody.innerHTML = data.runs.map((r) => `
    <tr>
      <td>${r.algorithm}</td>
      <td>${r.num_episodes}</td>
      <td>${fmt(r.final_avg_reward, 2)}</td>
      <td>${fmt(r.duration_seconds, 2)}s</td>
      <td><button class="btn" onclick="activateRun(${r.id})">Use this policy</button></td>
    </tr>
  `).join("");
}

// activate selected trained policy
async function activateRun(runId) {
  await postJSON(`${API}/train/activate`, { run_id: runId });
  loadRunCurve(runId);
}

// load reward curve for a training run
async function loadRunCurve(runId) {
  const run = await getJSON(`${API}/train/run/${runId}`);
  const curve = JSON.parse(run.reward_curve_json || "[]");
  drawLineChart(document.getElementById("chart-training-curve"), curve, { color: "#4f8cff" });
}

// load algorithm comparison data
async function loadCompareTable() {
  const data = await getJSON(`${API}/algorithms/compare`);
  const tbody = document.querySelector("#compare-table tbody");
  const typeTagClass = (t) => (t.includes("Dynamic") ? "dp" : t.includes("Monte") ? "mc" : "td");
  tbody.innerHTML = data.algorithms.map((a) => `
    <tr>
      <td>${a.label}</td>
      <td><span class="tag ${typeTagClass(a.type)}">${a.type}</span></td>
      <td>${a.model_required ? "Yes" : "No"}</td>
      <td>${a.learning_style}</td>
      <td>${a.on_off_policy}</td>
      <td>${a.has_results ? fmt(a.final_avg_reward, 2) : '<span class="no-results">not trained yet</span>'}</td>
      <td>${a.has_results ? fmt(a.duration_seconds, 2) + "s" : "-"}</td>
    </tr>
  `).join("");
}

// start dashboard updates
pollState();
pollTimer = setInterval(pollState, 800);
window.addEventListener("resize", () => { if (latestState) renderDashboard(latestState); }); 