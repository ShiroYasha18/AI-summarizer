const els = {
  mode: document.getElementById("mode"),
  apiBase: document.getElementById("apiBase"),
  saveApi: document.getElementById("saveApi"),
  doctorOverview: document.getElementById("doctorOverview"),
  refreshOverview: document.getElementById("refreshOverview"),
  overviewTable: document.getElementById("overviewTable"),
  title: document.getElementById("title"),
  patient: document.getElementById("patient"),
  range: document.getElementById("range"),
  start: document.getElementById("start"),
  end: document.getElementById("end"),
  startWrap: document.getElementById("startWrap"),
  endWrap: document.getElementById("endWrap"),
  simplifyWrap: document.getElementById("simplifyWrap"),
  simplifyNotes: document.getElementById("simplifyNotes"),
  load: document.getElementById("load"),
  meta: document.getElementById("meta"),
  risk: document.getElementById("risk"),
  metrics: document.getElementById("metrics"),
  notesMeta: document.getElementById("notesMeta"),
  notes: document.getElementById("notes"),
  genAi: document.getElementById("genAi"),
  aiSummary: document.getElementById("aiSummary"),
  rxMeta: document.getElementById("rxMeta"),
  rx: document.getElementById("rx"),
  labsMeta: document.getElementById("labsMeta"),
  labs: document.getElementById("labs"),
  loader: document.getElementById("loader"),
  loaderText: document.getElementById("loaderText"),
};

let charts = {
  bp: null,
  hrSpo2: null,
  temp: null,
  weightBmi: null,
};

function autoApiBase() {
  return "";
}

function apiBase() {
  const storedRaw = localStorage.getItem("apiBase");
  const stored = (storedRaw || "").trim().replace(/\/+$/, "");
  const { hostname } = window.location;
  const isLocal = hostname === "localhost" || hostname === "127.0.0.1";
  if (isLocal && /:8080$/.test(stored)) {
    localStorage.removeItem("apiBase");
    return "";
  }
  return (stored || autoApiBase()).replace(/\/+$/, "");
}

function setApiBase(v) {
  const val = (v || "").trim().replace(/\/+$/, "");
  if (val) localStorage.setItem("apiBase", val);
  else localStorage.removeItem("apiBase");
  els.apiBase.value = localStorage.getItem("apiBase") || "";
}

function fmt(v, decimals = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return n.toFixed(decimals);
}

function deltaStr(a, b, decimals = 0) {
  if (a === null || a === undefined || b === null || b === undefined) return "";
  const d = Number(b) - Number(a);
  if (!Number.isFinite(d)) return "";
  const sign = d > 0 ? "+" : "";
  return `${sign}${d.toFixed(decimals)}`;
}

function byId(id) {
  return document.getElementById(id);
}

function setBusy(on, text) {
  setHidden(els.loader, !on);
  if (els.loaderText) els.loaderText.textContent = text || "Loading…";
  els.load.disabled = Boolean(on);
  els.genAi.disabled = Boolean(on);
  els.refreshOverview.disabled = Boolean(on);
  els.mode.disabled = Boolean(on);
  els.patient.disabled = Boolean(on);
  els.range.disabled = Boolean(on);
  els.start.disabled = Boolean(on);
  els.end.disabled = Boolean(on);
  els.simplifyNotes.disabled = Boolean(on);
  els.saveApi.disabled = Boolean(on);
  els.apiBase.disabled = Boolean(on);
}

async function apiGet(path) {
  const base = apiBase();
  const res = await fetch(`${base}${path}`, { method: "GET" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiPost(path, body) {
  const base = apiBase();
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function setHidden(el, hidden) {
  if (!el) return;
  el.classList.toggle("hidden", Boolean(hidden));
}

function clearNode(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function parseIsoDate(d) {
  const s = String(d || "").slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return null;
  return s;
}

function setRangeFromPreset(checkins, days) {
  if (!checkins.length) return;
  const end = parseIsoDate(checkins[checkins.length - 1].date);
  if (!end) return;
  const endDt = new Date(`${end}T00:00:00Z`);
  const startDt = new Date(endDt.getTime() - (Math.max(0, days - 1) * 86400_000));
  const start = startDt.toISOString().slice(0, 10);
  els.start.value = start;
  els.end.value = end;
}

function destroyCharts() {
  Object.values(charts).forEach((c) => c && c.destroy());
  charts = { bp: null, hrSpo2: null, temp: null, weightBmi: null };
}

function buildLineDataset(label, data, color, dashed = false) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    tension: 0.25,
    pointRadius: 3,
    pointHoverRadius: 4,
    borderDash: dashed ? [6, 5] : [],
    fill: false,
  };
}

function buildBarDataset(label, data, color, dashed = false) {
  return {
    type: dashed ? "line" : "bar",
    label,
    data,
    borderColor: color,
    backgroundColor: dashed ? color : `${color}66`,
    pointRadius: dashed ? 0 : 0,
    borderDash: dashed ? [6, 5] : [],
    borderWidth: dashed ? 2 : 1,
  };
}

function createChart(canvasId, config) {
  const canvas = byId(canvasId);
  if (!canvas) return null;
  return new Chart(canvas, config);
}

function renderRisk(alerts) {
  clearNode(els.risk);
  if (!alerts || !alerts.length) return;
  alerts.forEach((a) => {
    const div = document.createElement("div");
    div.className = `alert ${a.severity || "medium"}`;
    const title = document.createElement("div");
    title.className = "alertTitle";
    title.textContent = a.headline || "Risk alert";
    const action = document.createElement("div");
    action.className = "alertAction";
    action.textContent = a.action || "";
    div.appendChild(title);
    div.appendChild(action);
    els.risk.appendChild(div);
  });
}

function renderMetricCards(payload) {
  clearNode(els.metrics);
  if (!payload || !payload.checkins || !payload.checkins.length) return;
  const rows = payload.checkins;
  const first = rows[0] || {};
  const last = rows[rows.length - 1] || {};

  const cards = [
    {
      label: "Blood pressure",
      value: `${fmt(last.SBP, 0)}/${fmt(last.DBP, 0)} mmHg`,
      delta: `${deltaStr(first.SBP, last.SBP, 0)}/${deltaStr(first.DBP, last.DBP, 0)}`,
      dir: (Number(last.SBP) - Number(first.SBP)) >= 0 ? "up" : "down",
    },
    {
      label: "Heart rate",
      value: `${fmt(last.HR, 0)} bpm`,
      delta: deltaStr(first.HR, last.HR, 0),
      dir: (Number(last.HR) - Number(first.HR)) >= 0 ? "up" : "down",
    },
    {
      label: "Oxygen",
      value: `${fmt(last.SpO2, 0)}%`,
      delta: deltaStr(first.SpO2, last.SpO2, 0),
      dir: (Number(last.SpO2) - Number(first.SpO2)) >= 0 ? "up" : "down",
    },
    {
      label: "Temperature",
      value: `${fmt(last.Temp, 1)} °C`,
      delta: deltaStr(first.Temp, last.Temp, 1),
      dir: (Number(last.Temp) - Number(first.Temp)) >= 0 ? "up" : "down",
    },
    {
      label: "Weight",
      value: `${fmt(last.weightKg, 1)} kg`,
      delta: deltaStr(first.weightKg, last.weightKg, 1),
      dir: (Number(last.weightKg) - Number(first.weightKg)) >= 0 ? "up" : "down",
    },
  ];

  cards.forEach((c) => {
    const card = document.createElement("div");
    card.className = "card";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = c.label;
    const value = document.createElement("div");
    value.className = "value";
    value.textContent = c.value;
    const delta = document.createElement("div");
    delta.className = `delta ${c.dir}`;
    delta.textContent = c.delta ? `${c.delta} vs baseline` : "";
    card.appendChild(label);
    card.appendChild(value);
    card.appendChild(delta);
    els.metrics.appendChild(card);
  });
}

function renderNotes(payload) {
  clearNode(els.notes);
  const meta = payload?.notes_meta;
  if (meta && meta.days) {
    els.notesMeta.textContent = `Last ${meta.days} days (${meta.start} → ${meta.end})`;
  } else {
    els.notesMeta.textContent = "";
  }
  const notes = payload?.notes || [];
  notes.forEach((n) => {
    const card = document.createElement("div");
    card.className = "note";
    const top = document.createElement("div");
    top.className = "noteTop";
    const dt = document.createElement("div");
    dt.className = "noteDate";
    dt.textContent = n.date || "";
    top.appendChild(dt);
    card.appendChild(top);
    if (n.glance) {
      const gl = document.createElement("div");
      gl.className = "noteGlance";
      gl.textContent = n.glance;
      card.appendChild(gl);
    }
    if (n.keywords && n.keywords.length) {
      const kw = document.createElement("div");
      kw.className = "noteKeywords";
      kw.textContent = `Keywords: ${n.keywords.join(" · ")}`;
      card.appendChild(kw);
    }
    const text = document.createElement("div");
    text.className = "noteText";
    text.textContent = n.text || "";
    card.appendChild(text);
    els.notes.appendChild(card);
  });
}

function renderGroupList(container, title, items) {
  const card = document.createElement("div");
  card.className = "note";
  const top = document.createElement("div");
  top.className = "noteTop";
  const t = document.createElement("div");
  t.className = "noteDate";
  t.textContent = title;
  top.appendChild(t);
  card.appendChild(top);
  const text = document.createElement("div");
  text.className = "noteText";
  text.style.whiteSpace = "pre-wrap";
  if (!items.length) text.textContent = "—";
  else text.textContent = items.map((x) => `• ${x}`).join("\n");
  card.appendChild(text);
  container.appendChild(card);
}

function renderRxLabs(payload) {
  clearNode(els.rx);
  clearNode(els.labs);
  if (els.rxMeta) els.rxMeta.textContent = "";
  if (els.labsMeta) els.labsMeta.textContent = "";

  const rx = payload?.rx || {};
  const labs = payload?.labs || {};

  const active = Array.isArray(rx.active_medications) ? rx.active_medications : [];
  const inactive = Array.isArray(rx.inactive_medications) ? rx.inactive_medications : [];
  const abn = Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs : [];
  const resolved = Array.isArray(labs.resolved_labs) ? labs.resolved_labs : [];

  const rxAsOf = String(rx.as_of_date || "").trim();
  const labsAsOf = String(labs.as_of_date || "").trim();

  if (els.rxMeta) els.rxMeta.textContent = `${rxAsOf ? `As of ${rxAsOf}. ` : ""}Active ${active.length}, inactive ${inactive.length}.`;
  if (els.labsMeta) els.labsMeta.textContent = `${labsAsOf ? `As of ${labsAsOf}. ` : ""}Abnormal ${abn.length}, resolved ${resolved.length}.`;

  renderGroupList(els.rx, "Active", active);
  renderGroupList(els.rx, "Inactive", inactive);
  renderGroupList(els.labs, "Abnormal", abn);
  renderGroupList(els.labs, "Resolved", resolved);
}

function renderCharts(payload) {
  destroyCharts();
  const rows = payload?.checkins || [];
  if (!rows.length) return;
  const labels = rows.map((r) => r.date);
  const baseline = payload?.baseline || {};

  const SBP = rows.map((r) => r.SBP);
  const DBP = rows.map((r) => r.DBP);
  const HR = rows.map((r) => r.HR);
  const SpO2 = rows.map((r) => r.SpO2);
  const Temp = rows.map((r) => r.Temp);
  const W = rows.map((r) => r.weightKg);
  const BMI = rows.map((r) => r.BMI);

  const sbpBase = labels.map(() => baseline.SBP ?? null);
  const dbpBase = labels.map(() => baseline.DBP ?? null);
  const hrBase = labels.map(() => baseline.HR ?? null);
  const spo2Base = labels.map(() => baseline.SpO2 ?? null);
  const tempBase = labels.map(() => baseline.Temp ?? null);
  const wBase = labels.map(() => baseline.weightKg ?? null);
  const bmiBase = labels.map(() => baseline.BMI ?? null);

  charts.bp = createChart("chartBp", {
    type: "line",
    data: {
      labels,
      datasets: [
        buildLineDataset("Systolic BP", SBP, "#8ab4f8"),
        buildLineDataset("Baseline", sbpBase, "#8ab4f8", true),
        buildLineDataset("Diastolic BP", DBP, "#1db954"),
        buildLineDataset("Baseline", dbpBase, "#1db954", true),
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: true } } },
  });

  charts.hrSpo2 = createChart("chartHrSpo2", {
    type: "line",
    data: {
      labels,
      datasets: [
        buildLineDataset("Heart rate", HR, "#fbbc04"),
        buildLineDataset("Baseline", hrBase, "#fbbc04", true),
        buildLineDataset("SpO2", SpO2, "#8ab4f8"),
        buildLineDataset("Baseline", spo2Base, "#8ab4f8", true),
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: true } } },
  });

  charts.temp = createChart("chartTemp", {
    type: "line",
    data: {
      labels,
      datasets: [buildLineDataset("Temperature", Temp, "#ea4335"), buildLineDataset("Baseline", tempBase, "#9aa0a6", true)],
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: true } } },
  });

  charts.weightBmi = createChart("chartWeightBmi", {
    type: "bar",
    data: {
      labels,
      datasets: [
        buildBarDataset("Weight (kg)", W, "#8ab4f8"),
        buildBarDataset("Baseline", wBase, "#8ab4f8", true),
        buildLineDataset("BMI", BMI, "#1db954"),
        buildLineDataset("Baseline", bmiBase, "#1db954", true),
      ],
    },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: true } } },
  });
}

function computeDateDefaults(checkins) {
  if (!checkins.length) return;
  const end = parseIsoDate(checkins[checkins.length - 1].date);
  if (!end) return;
  els.end.value = end;
  setRangeFromPreset(checkins, Number(els.range.value) || 30);
}

function setMode(mode) {
  const isDoctor = mode === "doctor";
  els.title.textContent = isDoctor ? "Doctor Dashboard" : "Patient Dashboard";
  setHidden(els.doctorOverview, !isDoctor);
  setHidden(els.simplifyWrap, !isDoctor);
}

async function loadPatients() {
  setBusy(true, "Loading patients…");
  try {
    const data = await apiGet("/api/patients");
    clearNode(els.patient);
    (data.patients || []).forEach((pid) => {
      const opt = document.createElement("option");
      opt.value = pid;
      opt.textContent = pid;
      els.patient.appendChild(opt);
    });
  } finally {
    setBusy(false);
  }
}

async function loadOverview() {
  setBusy(true, "Loading overview…");
  try {
    const data = await apiGet("/api/doctor/overview");
    const tbody = els.overviewTable.querySelector("tbody");
    clearNode(tbody);
    (data.rows || []).forEach((r) => {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td");
      td1.textContent = r.patient_id || "";
      const td2 = document.createElement("td");
      td2.textContent = r.last_checkin || "";
      const td3 = document.createElement("td");
      td3.textContent = r.risk || "";
      const td4 = document.createElement("td");
      td4.textContent = String(r.active_meds ?? "");
      const td5 = document.createElement("td");
      td5.textContent = String(r.abnormal_labs ?? "");
      tr.appendChild(td1);
      tr.appendChild(td2);
      tr.appendChild(td3);
      tr.appendChild(td4);
      tr.appendChild(td5);
      tr.addEventListener("click", () => {
        els.patient.value = r.patient_id;
        els.load.click();
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      tbody.appendChild(tr);
    });
  } finally {
    setBusy(false);
  }
}

async function loadDashboard() {
  setBusy(true, "Loading dashboard…");
  try {
    els.aiSummary.textContent = "";
    const role = els.mode.value;
    const patientId = els.patient.value;
    if (!patientId) return;
    const start = els.start.value;
    const end = els.end.value;
    const simplify = role === "doctor" && els.simplifyNotes.checked ? "&simplify_notes=1" : "";
    const payload = await apiGet(
      `/api/patients/${encodeURIComponent(patientId)}/dashboard?role=${role}&start=${start}&end=${end}${simplify}`
    );
    els.meta.textContent = `Showing ${payload.start_date} → ${payload.end_date}. Deltas are vs baseline (first day).`;
    renderRisk(payload.risk_alerts);
    renderMetricCards(payload);
    renderRxLabs(payload);
    renderCharts(payload);
    renderNotes(payload);
  } finally {
    setBusy(false);
  }
}

async function genAiSummary() {
  setBusy(true, "Generating AI summary…");
  try {
    const role = els.mode.value;
    const patientId = els.patient.value;
    const start = els.start.value;
    const end = els.end.value;
    if (!patientId) return;
    els.aiSummary.textContent = "Loading...";
    const out = await apiPost(`/api/patients/${encodeURIComponent(patientId)}/ai-summary`, {
      role,
      start_date: start,
      end_date: end,
      pipeline: "123",
    });
    els.aiSummary.textContent = out.text || "";
  } catch (e) {
    els.aiSummary.textContent = String(e?.message || e || "Failed");
  } finally {
    setBusy(false);
  }
}

async function bootstrap() {
  els.apiBase.value = localStorage.getItem("apiBase") || "";
  setMode(els.mode.value);
  const isCustom = els.range.value === "custom";
  setHidden(els.startWrap, !isCustom);
  setHidden(els.endWrap, !isCustom);

  try {
    await loadPatients();
  } catch (e) {
    els.meta.textContent = `API error: ${String(e?.message || e)}`;
    return;
  }

  const firstPid = els.patient.value;
  if (firstPid) {
    try {
      const dash = await apiGet(`/api/patients/${encodeURIComponent(firstPid)}/dashboard?role=patient`);
      const checkins = dash.checkins || [];
      computeDateDefaults(checkins);
    } catch {
      els.start.value = "";
      els.end.value = "";
    }
  }

  if (els.mode.value === "doctor") {
    loadOverview().catch(() => {});
  }
}

els.saveApi.addEventListener("click", () => {
  setApiBase(els.apiBase.value);
  bootstrap();
});

els.mode.addEventListener("change", () => {
  setMode(els.mode.value);
  if (els.mode.value === "doctor") {
    loadOverview().catch(() => {});
  }
});

els.range.addEventListener("change", async () => {
  const v = els.range.value;
  const isCustom = v === "custom";
  setHidden(els.startWrap, !isCustom);
  setHidden(els.endWrap, !isCustom);
  if (!isCustom) {
    try {
      const pid = els.patient.value;
      const dash = await apiGet(`/api/patients/${encodeURIComponent(pid)}/dashboard?role=${els.mode.value}`);
      setRangeFromPreset(dash.checkins || [], Number(v));
    } catch {}
  }
});

els.patient.addEventListener("change", async () => {
  try {
    const pid = els.patient.value;
    const dash = await apiGet(`/api/patients/${encodeURIComponent(pid)}/dashboard?role=${els.mode.value}`);
    computeDateDefaults(dash.checkins || []);
  } catch {}
});

els.refreshOverview.addEventListener("click", () => loadOverview().catch(() => {}));
els.load.addEventListener("click", () => loadDashboard().catch((e) => (els.meta.textContent = String(e?.message || e))));
els.genAi.addEventListener("click", () => genAiSummary());

bootstrap();
