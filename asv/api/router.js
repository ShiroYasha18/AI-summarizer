const admin = require("firebase-admin");

const METRICS = {
  SBP: { label: "Systolic BP", unit: "mmHg", meaning: "Blood pressure when the heart beats." },
  DBP: { label: "Diastolic BP", unit: "mmHg", meaning: "Blood pressure when the heart rests." },
  HR: { label: "Heart rate", unit: "bpm", meaning: "Heart beats per minute." },
  RR: { label: "Respiratory rate", unit: "breaths/min", meaning: "Breaths per minute." },
  SpO2: { label: "Oxygen saturation", unit: "%", meaning: "Oxygen level in blood." },
  Temp: { label: "Temperature", unit: "°C", meaning: "Body temperature." },
  Pulse: { label: "Pulse", unit: "bpm", meaning: "Pulse rate (may match HR)." },
  weightKg: { label: "Weight", unit: "kg", meaning: "Body weight." },
  BMI: { label: "BMI", unit: "", meaning: "Body mass index (weight vs height)." }
};

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
}

function parseFloatSafe(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const s = String(v).trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function parseBp(bp) {
  try {
    const s = String(bp || "");
    const parts = s.split("/", 2);
    if (parts.length !== 2) return [null, null];
    const a = Number(parts[0].trim());
    const b = Number(parts[1].trim());
    return [Number.isFinite(a) ? a : null, Number.isFinite(b) ? b : null];
  } catch {
    return [null, null];
  }
}

function isoDate10(v) {
  const s = String(v || "").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : "";
}

function series(values) {
  const out = [];
  for (const v of values) if (typeof v === "number" && Number.isFinite(v)) out.push(v);
  return out;
}

function statsForPeriod(rows) {
  if (!rows || !rows.length) return {};
  const metrics = ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"];
  const out = { days: rows.length, metrics: {} };
  for (const m of metrics) {
    const vals = series(rows.map((r) => r[m]));
    if (!vals.length) continue;
    out.metrics[m] = { min: Math.min(...vals), max: Math.max(...vals), avg: vals.reduce((a, b) => a + b, 0) / vals.length, last: vals[vals.length - 1] };
  }
  return out;
}

function baselineMap(rows, metrics) {
  if (!rows || !rows.length) return {};
  const first = rows[0] || {};
  const base = {};
  for (const m of metrics) {
    const v = first[m];
    if (typeof v === "number" && Number.isFinite(v)) base[m] = v;
  }
  return base;
}

function filterRange(items, start, end) {
  return (items || []).filter((x) => start <= String(x.date || "") && String(x.date || "") <= end);
}

function extractKeywords(text) {
  const t = String(text || "").trim().toLowerCase().split(/\s+/).join(" ");
  if (!t) return [];
  const tokens = t.match(/[a-z0-9%/]+/g) || [];
  const stop = new Set(["patient", "pt", "w", "with", "and", "the", "is", "are", "to", "of", "bp", "hr", "rr", "spo2", "temp", "stable", "continue", "current", "management", "plan", "monitor"]);
  const out = [];
  const seen = new Set();
  for (const tok of tokens) {
    if (stop.has(tok)) continue;
    if (tok.length < 3) continue;
    if (seen.has(tok)) continue;
    seen.add(tok);
    out.push(tok);
    if (out.length >= 6) break;
  }
  return out;
}

function noteGlance(text) {
  const t = String(text || "").trim();
  if (!t) return "";
  const parts = [];
  let m = t.match(/\bBP[:\s]*([0-9]{2,3})\s*\/\s*([0-9]{2,3})\b/i);
  if (m) parts.push(`BP ${m[1]}/${m[2]}`);
  m = t.match(/\bHR[:\s]*([0-9]{2,3})\b/i);
  if (m) parts.push(`HR ${m[1]}`);
  m = t.match(/\bRR[:\s]*([0-9]{1,3})\b/i);
  if (m) parts.push(`RR ${m[1]}`);
  m = t.match(/\bSpO2[:\s]*([0-9]{2,3})\s*%?\b/i);
  if (m) parts.push(`SpO2 ${m[1]}%`);
  m = t.match(/\bTemp(?:erature)?[:\s]*([0-9]{2,3}(?:\.[0-9])?)\b/i);
  if (m) parts.push(`Temp ${m[1]}°C`);
  return parts.join(" • ");
}

function notesFromRows(rows, limit) {
  const out = [];
  for (let i = rows.length - 1; i >= 0; i--) {
    const row = rows[i] || {};
    const s = String(row.notes_short || "").trim() || String(row.notes_full || "").trim();
    if (!s) continue;
    const body = s.split(/\s+/).join(" ").trim();
    out.push({
      date: row.date || "",
      glance: noteGlance(body),
      keywords: extractKeywords(body),
      text: body.length > 120 ? `${body.slice(0, 117).trimEnd()}...` : body,
      raw: body
    });
    if (out.length >= limit) break;
  }
  return out;
}

function riskTopic(flag) {
  const f = String(flag || "").trim().toLowerCase().split(/\s+/).join(" ");
  if (!f) return null;
  if (f.includes("stroke") || f.includes("tia")) return "stroke";
  if (f.includes("heart attack") || f.includes(" mi ") || f.includes("myocard")) return "heart attack";
  if (f.includes("bleed") || f.includes("hemorr") || f.includes("haemorr")) return "bleeding";
  if (f.includes("sepsis")) return "sepsis";
  if (f.includes("kidney") || f.includes("renal") || f.includes("aki")) return "kidney injury";
  if (f.includes("hyperkal") || f.includes("potassium high")) return "high potassium";
  if (f.includes("hypogly") || f.includes("glucose low") || f.includes("low sugar")) return "low blood sugar";
  if (f.includes("fall") || f.includes("faint") || f.includes("syncope")) return "fainting";
  return null;
}

function medCount(rx) {
  const active = rx && Array.isArray(rx.active_medications) ? rx.active_medications.length : null;
  return typeof active === "number" && Number.isFinite(active) ? active : null;
}

function filterRiskFlags(flags, rx) {
  const meds = medCount(rx);
  const out = [];
  for (const f of Array.isArray(flags) ? flags : []) {
    const s = String(f || "").trim();
    if (!s) continue;
    const low = s.toLowerCase();
    if (meds !== null && meds < 5 && low.includes("polypharmacy")) continue;
    out.push(s);
  }
  return out;
}

function riskAlerts(rows, rx, labs) {
  const alerts = [];
  const metrics = (rows && rows.length) ? (statsForPeriod(rows).metrics || {}) : {};
  const sbpMax = metrics.SBP?.max;
  const dbpMax = metrics.DBP?.max;
  const spo2Min = metrics.SpO2?.min;
  const tempMax = metrics.Temp?.max;
  const hrMax = metrics.HR?.max;
  const hrMin = metrics.HR?.min;

  if (sbpMax !== undefined && dbpMax !== undefined && (sbpMax >= 180 || dbpMax >= 120)) {
    alerts.push({ severity: "critical", headline: "High risk of hypertensive crisis", action: "Talk to a doctor immediately (or seek urgent care)." });
  } else if (sbpMax !== undefined && dbpMax !== undefined && (sbpMax >= 160 || dbpMax >= 100)) {
    alerts.push({ severity: "high", headline: "High blood pressure", action: "Talk to your doctor soon." });
  }

  if (spo2Min !== undefined && spo2Min < 90) {
    alerts.push({ severity: "critical", headline: "High risk from low oxygen", action: "Talk to a doctor immediately (or seek urgent care)." });
  } else if (spo2Min !== undefined && spo2Min < 92) {
    alerts.push({ severity: "high", headline: "Low oxygen", action: "Talk to your doctor soon." });
  }

  if (tempMax !== undefined && tempMax >= 39.0) alerts.push({ severity: "high", headline: "High fever", action: "Talk to your doctor soon." });
  if (hrMax !== undefined && hrMax >= 130) alerts.push({ severity: "high", headline: "Very high heart rate", action: "Talk to your doctor soon." });
  if (hrMin !== undefined && hrMin <= 40) alerts.push({ severity: "high", headline: "Very low heart rate", action: "Talk to your doctor soon." });

  const flags = [];
  if (rx && Array.isArray(rx.risk_flags)) flags.push(...filterRiskFlags(rx.risk_flags, rx));
  if (labs && Array.isArray(labs.risk_flags)) flags.push(...labs.risk_flags);

  for (const f of flags) {
    const topic = riskTopic(f);
    if (topic) alerts.push({ severity: "high", headline: `High risk of ${topic}`, action: "Talk to your doctor soon." });
    else {
      const cleaned = String(f || "").trim().split(/\s+/).join(" ");
      if (cleaned) alerts.push({ severity: "medium", headline: `Risk flag: ${cleaned}`, action: "Discuss with your doctor." });
    }
  }

  const abn = labs && Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs : [];
  if (abn.length) {
    const names = [];
    for (const x of abn.slice(0, 6)) {
      if (typeof x === "string") names.push(x);
      else if (x && typeof x === "object") {
        const n = x.test_name || x.testName || x.name || x.test || x.lab || "";
        if (n) names.push(String(n));
      }
    }
    const msg = names.length ? `Abnormal labs: ${names.slice(0, 3).join(", ")}.` : "Abnormal lab results detected.";
    alerts.push({ severity: "medium", headline: msg, action: "Discuss with your doctor." });
  }

  const order = { critical: 3, high: 2, medium: 1, low: 0 };
  const seen = new Set();
  const uniq = [];
  for (const a of alerts) {
    const key = `${a.severity || ""}::${a.headline || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    uniq.push(a);
  }
  uniq.sort((a, b) => (order[b.severity] || 0) - (order[a.severity] || 0));
  return uniq.slice(0, 5);
}

function latestSummaryOnOrBefore(summaries, endDate) {
  let best = null;
  for (const s of summaries) {
    const dt = String(s.date || "");
    if (dt && dt <= endDate) best = s;
    else if (dt && dt > endDate) break;
  }
  return best || {};
}

function stringifyBrief(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) return v.map(stringifyBrief).filter(Boolean).join(", ");
  if (typeof v === "object") {
    const testName = v.test_name || v.testName || v.test || v.lab || v.name;
    if (testName) {
      const current =
        v.current_value ??
        v.currentValue ??
        v.value ??
        v.current ??
        v.result ??
        v.reading ??
        v.level ??
        null;
      const unit = v.unit || v.units || "";
      const resolved = isoDate10(v.resolution_date || v.resolutionDate || v.resolved_date || v.resolvedDate || v.date || "");
      const prevHigh = v.previous_high ?? v.previousHigh ?? null;
      const prevLow = v.previous_low ?? v.previousLow ?? null;
      const ref = v.reference_range || v.referenceRange || v.ref_range || v.refRange || "";
      const src = String(v.source_doc_id || v.sourceDocId || v.source || "").trim();

      const head = String(testName).trim();
      const val =
        current === null || current === undefined || current === ""
          ? ""
          : `${String(current).trim()}${unit ? ` ${String(unit).trim()}` : ""}`;
      const extras = [];
      if (resolved) extras.push(`resolved ${resolved}`);
      if (prevHigh !== null && prevHigh !== undefined && prevHigh !== "") extras.push(`prev high ${String(prevHigh).trim()}`);
      if (prevLow !== null && prevLow !== undefined && prevLow !== "") extras.push(`prev low ${String(prevLow).trim()}`);
      if (ref) extras.push(`ref ${String(ref).trim()}`);
      if (src) extras.push(src);

      const tail = extras.length ? ` (${extras.join(", ")})` : "";
      return `${head}${val ? `: ${val}` : ""}${tail}`;
    }

    const medName = v.medication || v.drug || v.name || v.title || v.item;
    if (medName) {
      const dose = v.dose || v.dosage || v.strength || v.amount || "";
      const freq = v.frequency || v.freq || v.schedule || "";
      const route = v.route || "";
      const parts = [];
      if (dose) parts.push(String(dose).trim());
      if (freq) parts.push(String(freq).trim());
      if (route) parts.push(String(route).trim());
      return `${String(medName).trim()}${parts.length ? ` — ${parts.join(" ")}` : ""}`;
    }

    try {
      const s = JSON.stringify(v);
      return s.length > 200 ? `${s.slice(0, 197)}...` : s;
    } catch {
      return "";
    }
  }
  return String(v);
}

function normalizeStringList(items, max) {
  const out = [];
  const seen = new Set();
  for (const it of Array.isArray(items) ? items : []) {
    const s = stringifyBrief(it).trim().split(/\s+/).join(" ");
    if (!s) continue;
    const key = s.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
    if (out.length >= max) break;
  }
  return out;
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (c) => {
      body += c;
      if (body.length > 1_000_000) {
        reject(new Error("Payload too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!body.trim()) return resolve({});
      try {
        resolve(JSON.parse(body));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function getServiceAccount() {
  const raw = String(process.env.FIREBASE_SERVICE_ACCOUNT_JSON || "").trim();
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function firestore() {
  if (admin.apps.length) return admin.firestore();
  const sa = getServiceAccount();
  if (!sa) throw new Error("Missing FIREBASE_SERVICE_ACCOUNT_JSON env var");
  admin.initializeApp({ credential: admin.credential.cert(sa) });
  return admin.firestore();
}

async function listPatients(db) {
  const snap = await db.collection("Patients").get();
  const out = snap.docs.map((d) => d.id).sort();
  return out;
}

function parseCheckinDoc(doc) {
  const d = doc.data() || {};
  const dt = isoDate10(d.date || d.Date || d.createdAt || "");
  const vitals = Array.isArray(d.vitals) ? d.vitals : [];
  const v0 = vitals.length && vitals[0] && typeof vitals[0] === "object" ? vitals[0] : {};
  const [sbp, dbp] = parseBp(v0.BP);
  const notesObj = d.Notes && typeof d.Notes === "object" ? d.Notes : {};
  return {
    date: dt,
    SBP: sbp,
    DBP: dbp,
    HR: parseFloatSafe(v0.HR),
    RR: parseFloatSafe(v0.RR),
    SpO2: parseFloatSafe(v0.SpO2),
    Temp: parseFloatSafe(v0.Temp),
    Pulse: parseFloatSafe(v0.Pulse),
    weightKg: parseFloatSafe(v0.weightKg),
    BMI: parseFloatSafe(v0.BMI),
    notes_short: String(notesObj.short || ""),
    notes_full: String(notesObj.full || "")
  };
}

function sortByDateAsc(rows) {
  rows.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  return rows;
}

async function loadDailyCheckinsAll(db, patientId) {
  const snap = await db.collection("Patients").doc(patientId).collection("DailyCheckIns").get();
  const rows = snap.docs.map(parseCheckinDoc).filter((x) => x.date);
  return sortByDateAsc(rows);
}

async function loadDailyCheckinsRecent(db, patientId, limit) {
  const col = db.collection("Patients").doc(patientId).collection("DailyCheckIns");
  let snap;
  try {
    snap = await col.orderBy("date", "desc").limit(Math.max(1, Number(limit) || 30)).get();
  } catch {
    return await loadDailyCheckinsAll(db, patientId);
  }
  let rows = snap.docs.map(parseCheckinDoc).filter((x) => x.date);
  if (!rows.length) {
    const probe = await col.limit(1).get();
    if (probe.empty) return [];
    const d = probe.docs[0].data() || {};
    if (!("date" in d)) return await loadDailyCheckinsAll(db, patientId);
  }
  rows = sortByDateAsc(rows);
  return rows;
}

async function loadDailyCheckinsRange(db, patientId, start, end) {
  const col = db.collection("Patients").doc(patientId).collection("DailyCheckIns");
  let q = col;
  if (start) q = q.where("date", ">=", start);
  if (end) q = q.where("date", "<=", end);
  let snap;
  try {
    snap = await q.orderBy("date", "asc").get();
  } catch {
    const all = await loadDailyCheckinsAll(db, patientId);
    return filterRange(all, start, end);
  }
  let rows = snap.docs.map(parseCheckinDoc).filter((x) => x.date);
  if (!rows.length) {
    const probe = await col.limit(1).get();
    if (probe.empty) return [];
    const d = probe.docs[0].data() || {};
    if (!("date" in d)) {
      const all = await loadDailyCheckinsAll(db, patientId);
      return filterRange(all, start, end);
    }
  }
  return rows;
}

async function loadSummaries(db, patientId, pipeline) {
  const snap = await db
    .collection("Summaries")
    .where("patientId", "==", patientId)
    .where("pipeline", "==", pipeline)
    .get();
  const out = [];
  for (const doc of snap.docs) {
    const d = doc.data() || {};
    const dt = isoDate10(d.date || "");
    if (!dt) continue;
    out.push({ ...d, date: dt, _doc_id: doc.id });
  }
  out.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  return out;
}

async function loadLatestSummary(db, patientId, pipeline) {
  try {
    const snap = await db
      .collection("Summaries")
      .where("patientId", "==", patientId)
      .where("pipeline", "==", pipeline)
      .orderBy("date", "desc")
      .limit(1)
      .get();
    const doc = snap.docs[0];
    if (!doc) return {};
    const d = doc.data() || {};
    const dt = isoDate10(d.date || "");
    return dt ? { ...d, date: dt, _doc_id: doc.id } : {};
  } catch {
    const all = await loadSummaries(db, patientId, pipeline);
    return all.length ? all[all.length - 1] : {};
  }
}

async function geminiGenerateText(prompt) {
  const key = String(process.env.GOOGLE_API_KEY || "").trim();
  if (!key) return "";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(key)}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ role: "user", parts: [{ text: prompt }] }] })
  });
  if (!resp.ok) return "";
  const data = await resp.json().catch(() => null);
  const text = data?.candidates?.[0]?.content?.parts?.map((p) => p.text).filter(Boolean).join("") || "";
  return String(text || "").trim();
}

function extractFirstJsonObject(text) {
  const s = String(text || "");
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  const candidate = s.slice(start, end + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

function extractFirstJsonValue(text) {
  const s = String(text || "");
  const candidates = [];
  const oStart = s.indexOf("{");
  const oEnd = s.lastIndexOf("}");
  if (oStart !== -1 && oEnd !== -1 && oEnd > oStart) candidates.push({ start: oStart, end: oEnd });
  const aStart = s.indexOf("[");
  const aEnd = s.lastIndexOf("]");
  if (aStart !== -1 && aEnd !== -1 && aEnd > aStart) candidates.push({ start: aStart, end: aEnd });
  candidates.sort((a, b) => a.start - b.start);

  for (const c of candidates) {
    const candidate = s.slice(c.start, c.end + 1);
    try {
      return JSON.parse(candidate);
    } catch {}
  }
  return null;
}

async function aiSimpleNotes(notes) {
  const items = notes.map((n) => ({ date: String(n.date || ""), note: String(n.raw || "").slice(0, 500) }));
  const prompt =
    "Rewrite each clinical note in simple English for quick reading.\n" +
    "Rules:\n" +
    "- One short sentence per note (max 14 words)\n" +
    "- Keep key vitals if present (BP/HR/SpO2/Temp)\n" +
    "- Keep the main issue (e.g., BP elevated, stable, improving)\n" +
    "- Output ONLY valid JSON: {\"YYYY-MM-DD\": \"sentence\", ...}\n\n" +
    `JSON:\n${JSON.stringify({ notes: items }, null, 2)}`;
  const text = await geminiGenerateText(prompt);
  const obj = extractFirstJsonObject(text);
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return {};
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    const kk = String(k || "").trim().slice(0, 10);
    const vv = String(v || "").trim().split(/\s+/).join(" ");
    if (kk && vv) out[kk] = vv;
  }
  return out;
}

async function aiSummary(role, patientId, startDate, endDate, vitalsStats, rx, labs) {
  const payload = {
    role,
    patient_id: patientId,
    start_date: startDate,
    end_date: endDate,
    vitals_stats: vitalsStats,
    prescriptions_asof: rx,
    labs_asof: labs
  };
  const prompt =
    (role === "patient"
      ? "You are a patient-friendly health dashboard assistant.\nSummarize the selected period in simple language.\nFocus on: (1) what changed, (2) what to watch, (3) questions to ask the doctor.\nKeep it short and easy to scan. Use bullet points.\n\n"
      : "You are a clinician-facing dashboard assistant.\nSummarize the selected period for rapid review.\nFocus on: changes vs baseline, potential concerns, medication/lab deltas, and next actions/questions.\nUse concise bullet points.\n\n") +
    `JSON:\n${JSON.stringify(payload, null, 2)}`;
  return await geminiGenerateText(prompt);
}

function safeText(v, maxLen) {
  const s = String(v || "").trim().split(/\s+/).join(" ");
  if (!s) return "";
  const n = Number(maxLen) || 240;
  return s.length > n ? `${s.slice(0, Math.max(0, n - 3)).trimEnd()}...` : s;
}

function fmtNum(v, decimals) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(Number(decimals) || 0);
}

function toInt(v, def) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : def;
}

function normalizeAudience(v) {
  const s = String(v || "").trim().toLowerCase();
  if (s === "clinic" || s === "clinician" || s === "doctor") return "clinic";
  if (s === "both") return "both";
  return "patient";
}

function normalizeTask(input, idx) {
  const t = input && typeof input === "object" ? input : {};
  const title = safeText(t.title || t.task || t.name || "", 120);
  if (!title) return null;
  const audience = normalizeAudience(t.audience || t.for || t.owner);
  const typeRaw = String(t.type || t.schedule_type || (t.schedule && t.schedule.type) || "").trim().toLowerCase();
  const type = typeRaw === "once" || typeRaw === "daily" || typeRaw === "conditional" ? typeRaw : "once";
  const schedule = t.schedule && typeof t.schedule === "object" ? t.schedule : {};
  const startDate = isoDate10(schedule.start_date || schedule.start || t.start_date || "");
  const endDate = isoDate10(schedule.end_date || schedule.end || t.end_date || "");
  const dueDate = isoDate10(schedule.date || schedule.due_date || t.due_date || "");
  const days = Math.max(1, toInt(schedule.days || t.days, 1));
  const condition = safeText(t.condition || schedule.condition || "", 200);
  const explanation = safeText(t.explanation || t.reason || "", 260);
  const action = safeText(t.action || t.instructions || "", 260);
  return {
    id: safeText(t.id || `${idx + 1}`, 48),
    audience,
    title,
    type,
    schedule: {
      start_date: startDate,
      end_date: endDate,
      date: dueDate,
      days: type === "daily" ? days : undefined
    },
    condition: type === "conditional" ? condition : "",
    action,
    explanation
  };
}

async function aiActionTasks(role, patientId, startDate, endDate, vitalsStats, notes, rx, labs) {
  const key = String(process.env.GOOGLE_API_KEY || "").trim();
  if (!key) return { tasks: [], warning: "GOOGLE_API_KEY not set" };
  const notesIn = (Array.isArray(notes) ? notes : []).slice(0, 12).map((n) => ({ date: String(n.date || ""), note: String(n.raw || "").slice(0, 600) }));
  const payload = {
    role,
    patient_id: patientId,
    start_date: startDate,
    end_date: endDate,
    vitals_stats: vitalsStats,
    prescriptions_asof: rx,
    labs_asof: labs,
    notes: notesIn
  };
  const prompt =
    "Convert the notes and prescriptions into actionable tasks for patient and clinic.\n" +
    "Rules:\n" +
    "- Output ONLY valid JSON: {\"tasks\": [...]}\n" +
    "- Each task: {\"title\",\"audience\":\"patient|clinic|both\",\"type\":\"once|daily|conditional\",\"schedule\":{...},\"condition\",\"action\",\"explanation\"}\n" +
    "- For daily tasks: set type=\"daily\" and schedule.days=N (e.g., 3 days)\n" +
    "- For conditional tasks: type=\"conditional\" and include condition\n" +
    "- Keep <= 12 tasks. Titles short.\n\n" +
    `JSON:\n${JSON.stringify(payload, null, 2)}`;
  const text = await geminiGenerateText(prompt);
  const parsed = extractFirstJsonValue(text);
  const obj = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  const tasksRaw = obj && Array.isArray(obj.tasks) ? obj.tasks : [];
  const tasks = [];
  for (let i = 0; i < tasksRaw.length; i++) {
    const norm = normalizeTask(tasksRaw[i], i);
    if (norm) tasks.push(norm);
    if (tasks.length >= 12) break;
  }
  return { tasks };
}

function normalizeSeverity(v) {
  const s = String(v || "").trim().toLowerCase();
  if (s === "critical" || s === "high" || s === "medium" || s === "low") return s;
  return "medium";
}

function normalizeSignal(v) {
  const s = String(v || "").trim().toLowerCase();
  const allowed = new Set([
    "hypertensive_crisis",
    "high_bp_with_symptoms",
    "low_oxygen",
    "high_fever",
    "tachycardia_with_symptoms",
    "bradycardia_with_symptoms",
    "chest_pain",
    "medication_or_lab_risk",
    "other"
  ]);
  return allowed.has(s) ? s : "other";
}

function normalizeAiAlert(a) {
  const o = a && typeof a === "object" ? a : {};
  const headline = safeText(o.headline || "", 120);
  if (!headline) return null;
  return {
    signal: normalizeSignal(o.signal),
    severity: normalizeSeverity(o.severity),
    headline,
    reason: safeText(o.reason || "", 240),
    action: safeText(o.action || "", 220),
    patient_explanation: safeText(o.patient_explanation || o.patientExplanation || "", 420),
    clinician_explanation: safeText(o.clinician_explanation || o.clinicianExplanation || "", 520)
  };
}

function evidenceForSignal(signal, rows, notes, rx, labs) {
  const ev = [];
  const meds = normalizeStringList(rx && Array.isArray(rx.active_medications) ? rx.active_medications : [], 12);
  const abnLabs = normalizeStringList(labs && Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs : [], 12);

  if (signal === "hypertensive_crisis") {
    ev.push(vitEvidence("Systolic BP", "SBP", "mmHg", ">= 180", pointsWhere(rows, "SBP", (v) => v >= 180, 3)));
    ev.push(vitEvidence("Diastolic BP", "DBP", "mmHg", ">= 120", pointsWhere(rows, "DBP", (v) => v >= 120, 3)));
  } else if (signal === "high_bp_with_symptoms") {
    ev.push(vitEvidence("Systolic BP", "SBP", "mmHg", ">= 160", pointsWhere(rows, "SBP", (v) => v >= 160, 3)));
    ev.push(noteEvidence("Symptoms mentioned", noteMatches(notes, ["headache", "migraine", "dizzy", "dizziness", "chest pain", "shortness of breath", "breathless"], 2)));
  } else if (signal === "low_oxygen") {
    ev.push(vitEvidence("Oxygen (SpO2)", "SpO2", "%", "< 92", pointsWhere(rows, "SpO2", (v) => v < 92, 3)));
    ev.push(noteEvidence("Breathing symptoms mentioned", noteMatches(notes, ["shortness of breath", "sob", "breathless", "difficulty breathing"], 2)));
  } else if (signal === "high_fever") {
    ev.push(vitEvidence("Temperature", "Temp", "°C", ">= 39.0", pointsWhere(rows, "Temp", (v) => v >= 39.0, 3)));
    ev.push(noteEvidence("Symptoms mentioned", noteMatches(notes, ["confusion", "confused", "breathless", "shortness of breath"], 2)));
  } else if (signal === "tachycardia_with_symptoms") {
    ev.push(vitEvidence("Heart rate", "HR", "bpm", ">= 130", pointsWhere(rows, "HR", (v) => v >= 130, 3)));
    ev.push(noteEvidence("Symptoms mentioned", noteMatches(notes, ["chest pain", "dizzy", "dizziness", "shortness of breath", "breathless"], 2)));
  } else if (signal === "bradycardia_with_symptoms") {
    ev.push(vitEvidence("Heart rate", "HR", "bpm", "<= 40", pointsWhere(rows, "HR", (v) => v <= 40, 3)));
    ev.push(noteEvidence("Symptoms mentioned", noteMatches(notes, ["dizzy", "dizziness", "faint", "syncope", "lightheaded"], 2)));
  } else if (signal === "chest_pain") {
    ev.push(noteEvidence("Chest pain mentioned", noteMatches(notes, ["chest pain", "chest tightness", "pressure in chest"], 2)));
  } else if (signal === "medication_or_lab_risk") {
    if (meds.length) ev.push(flagEvidence("meds", meds));
    if (abnLabs.length) ev.push(flagEvidence("abnormal_labs", abnLabs));
  } else if (signal === "other") {
    const bpHigh = pointsWhere(rows, "SBP", (v) => v >= 160, 3);
    const spo2Low = pointsWhere(rows, "SpO2", (v) => v < 92, 3);
    const tempHigh = pointsWhere(rows, "Temp", (v) => v >= 39.0, 3);
    const hrHigh = pointsWhere(rows, "HR", (v) => v >= 130, 3);
    const hrLow = pointsWhere(rows, "HR", (v) => v <= 40, 3);

    ev.push(vitEvidence("Systolic BP", "SBP", "mmHg", ">= 160", bpHigh));
    ev.push(vitEvidence("Oxygen (SpO2)", "SpO2", "%", "< 92", spo2Low));
    ev.push(vitEvidence("Temperature", "Temp", "°C", ">= 39.0", tempHigh));
    ev.push(vitEvidence("Heart rate", "HR", "bpm", ">= 130", hrHigh));
    ev.push(vitEvidence("Heart rate", "HR", "bpm", "<= 40", hrLow));

    ev.push(
      noteEvidence(
        "Symptoms mentioned",
        noteMatches(
          notes,
          [
            "headache",
            "migraine",
            "chest pain",
            "chest tightness",
            "pressure in chest",
            "shortness of breath",
            "sob",
            "breathless",
            "difficulty breathing",
            "dizzy",
            "dizziness",
            "lightheaded",
            "faint",
            "syncope",
            "confusion",
            "confused",
            "disoriented"
          ],
          2
        )
      )
    );

    if (meds.length) ev.push(flagEvidence("meds", meds));
    if (abnLabs.length) ev.push(flagEvidence("abnormal_labs", abnLabs));
  }

  return ev.filter(Boolean);
}

async function aiRiskAlerts(role, patientId, startDate, endDate, rows, notes, rx, labs) {
  const key = String(process.env.GOOGLE_API_KEY || "").trim();
  if (!key) return { alerts: [], warning: "GOOGLE_API_KEY not set" };

  const vitals = (Array.isArray(rows) ? rows : []).slice(-30).map((r) => ({
    date: String(r?.date || ""),
    SBP: r?.SBP ?? null,
    DBP: r?.DBP ?? null,
    HR: r?.HR ?? null,
    SpO2: r?.SpO2 ?? null,
    Temp: r?.Temp ?? null
  }));

  const symptoms = [
    ...noteMatches(notes, ["headache", "migraine"], 3),
    ...noteMatches(notes, ["chest pain", "chest tightness", "pressure in chest"], 3),
    ...noteMatches(notes, ["shortness of breath", "sob", "breathless", "difficulty breathing"], 3),
    ...noteMatches(notes, ["dizzy", "dizziness", "lightheaded", "faint", "syncope"], 3),
    ...noteMatches(notes, ["confusion", "confused", "disoriented"], 3)
  ].slice(0, 8);

  const meds = normalizeStringList(rx && Array.isArray(rx.active_medications) ? rx.active_medications : [], 18);
  const abnormalLabs = normalizeStringList(labs && Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs : [], 18);

  const observed = {
    sbp_max: maxPoint(rows, "SBP"),
    dbp_max: maxPoint(rows, "DBP"),
    spo2_min: minPoint(rows, "SpO2"),
    temp_max: maxPoint(rows, "Temp"),
    hr_max: maxPoint(rows, "HR"),
    hr_min: minPoint(rows, "HR")
  };

  const payload = {
    role,
    patient_id: patientId,
    start_date: startDate,
    end_date: endDate,
    vitals_last_30: vitals,
    observed_extremes: observed,
    symptoms_from_notes: symptoms,
    active_medications: meds,
    abnormal_labs: abnormalLabs
  };

  const prompt =
    "You are a clinical safety assistant.\n" +
    "Generate risk alerts ONLY based on the provided data. Do not invent dates, values, symptoms, meds, or labs.\n" +
    "Output ONLY valid JSON: {\"alerts\": [...]}\n" +
    "Each alert:\n" +
    "- signal: one of [hypertensive_crisis, high_bp_with_symptoms, low_oxygen, high_fever, tachycardia_with_symptoms, bradycardia_with_symptoms, chest_pain, medication_or_lab_risk, other]\n" +
    "- severity: critical|high|medium|low\n" +
    "- headline: short\n" +
    "- reason: one sentence\n" +
    "- action: one sentence\n" +
    "- patient_explanation: short, normal language\n" +
    "- clinician_explanation: short, clinical language\n" +
    "Rules:\n" +
    "- If there is no evidence for a risk, do not include that alert.\n" +
    "- Keep <= 6 alerts.\n\n" +
    `JSON:\n${JSON.stringify(payload, null, 2)}`;

  const text = await geminiGenerateText(prompt);
  if (!text) return { alerts: [], warning: "Gemini unavailable" };
  const parsed = extractFirstJsonValue(text);
  const obj = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  if (!obj) return { alerts: [], warning: "Invalid Gemini response" };
  const rawAlerts = obj && Array.isArray(obj.alerts) ? obj.alerts : [];

  const out = [];
  const medsCount = Array.isArray(meds) ? meds.length : 0;
  for (const a of rawAlerts) {
    const norm = normalizeAiAlert(a);
    if (!norm) continue;
    const evidence = evidenceForSignal(norm.signal, rows, notes, rx, labs);
    if (!evidence.length) continue;
    const polyText = `${norm.headline} ${norm.reason} ${norm.action} ${norm.patient_explanation} ${norm.clinician_explanation}`.toLowerCase();
    if (medsCount < 5 && polyText.includes("polypharmacy")) continue;
    out.push({
      severity: norm.severity,
      headline: norm.headline,
      reason: norm.reason,
      action: norm.action,
      evidence,
      explanation: role === "doctor" ? (norm.clinician_explanation || norm.reason) : (norm.patient_explanation || norm.reason)
    });
    if (out.length >= 6) break;
  }

  return { alerts: out, warning: "" };
}

function noteHasAny(notes, needles) {
  const hay = (Array.isArray(notes) ? notes : []).map((n) => String(n.raw || n.text || "")).join(" ").toLowerCase();
  if (!hay.trim()) return false;
  for (const w of needles) if (hay.includes(w)) return true;
  return false;
}

function noteMatches(notes, needles, limit) {
  const out = [];
  const uniq = new Set();
  const want = (Array.isArray(needles) ? needles : []).map((x) => String(x || "").trim().toLowerCase()).filter(Boolean);
  const max = Math.max(1, Number(limit) || 3);
  for (let i = (Array.isArray(notes) ? notes.length : 0) - 1; i >= 0; i--) {
    const n = notes[i] || {};
    const text = String(n.raw || n.text || "").toLowerCase();
    if (!text.trim()) continue;
    let hit = "";
    for (const w of want) {
      if (text.includes(w)) {
        hit = w;
        break;
      }
    }
    if (!hit) continue;
    const dt = isoDate10(n.date || "");
    const snippet = safeText(n.raw || n.text || "", 180);
    const key = `${dt}::${hit}::${snippet}`;
    if (uniq.has(key)) continue;
    uniq.add(key);
    out.push({ date: dt, keyword: hit, text: snippet });
    if (out.length >= max) break;
  }
  return out;
}

function finite(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function maxPoint(rows, field) {
  let best = null;
  for (const r of Array.isArray(rows) ? rows : []) {
    const v = r ? r[field] : null;
    if (!finite(v)) continue;
    if (!best || v > best.value) best = { date: String(r.date || ""), value: v };
  }
  return best;
}

function minPoint(rows, field) {
  let best = null;
  for (const r of Array.isArray(rows) ? rows : []) {
    const v = r ? r[field] : null;
    if (!finite(v)) continue;
    if (!best || v < best.value) best = { date: String(r.date || ""), value: v };
  }
  return best;
}

function pointsWhere(rows, field, predicate, limit) {
  const out = [];
  const max = Math.max(1, Number(limit) || 3);
  for (let i = (Array.isArray(rows) ? rows.length : 0) - 1; i >= 0; i--) {
    const r = rows[i] || {};
    const v = r[field];
    if (!finite(v)) continue;
    if (!predicate(v)) continue;
    out.push({ date: String(r.date || ""), value: v });
    if (out.length >= max) break;
  }
  return out.reverse();
}

function vitEvidence(label, metric, unit, threshold, points) {
  const pts = Array.isArray(points) ? points.filter((p) => p && p.date && finite(p.value)) : [];
  return pts.length ? { kind: "vital", label, metric, unit, threshold, points: pts } : null;
}

function noteEvidence(label, matches) {
  const pts = Array.isArray(matches) ? matches.filter((m) => m && m.text) : [];
  return pts.length ? { kind: "note", label, points: pts } : null;
}

function flagEvidence(source, flags) {
  const items = normalizeStringList(flags, 8);
  return items.length ? { kind: "flag", source, flags: items } : null;
}

function plainFlag(flag) {
  const raw = String(flag || "").trim();
  const s = raw.toLowerCase();
  if (!s) return "";
  if (s.includes("polypharmacy")) return "polypharmacy risk (many medications)";
  return raw.split("_").join(" ").split(/\s+/).join(" ").trim();
}

function explainAlert(role, headline, evidence) {
  const ev = Array.isArray(evidence) ? evidence : [];
  const parts = [];
  const vitals = ev.filter((e) => e && e.kind === "vital");
  const notes = ev.filter((e) => e && e.kind === "note");
  const flags = ev.filter((e) => e && e.kind === "flag");

  if (vitals.length) {
    const lines = [];
    for (const v of vitals) {
      const p0 = (v.points && v.points[0]) || null;
      if (!p0) continue;
      const day = p0.date ? ` on ${p0.date}` : "";
      const unit = v.unit ? ` ${v.unit}` : "";
      const thr = v.threshold ? ` (${v.threshold})` : "";
      lines.push(`${v.label}: ${fmtNum(p0.value, 0)}${unit}${day}${thr}`);
      if (lines.length >= 3) break;
    }
    if (lines.length) parts.push(lines.join(role === "doctor" ? "; " : ". "));
  }

  if (notes.length) {
    const n0 = (notes[0].points && notes[0].points[0]) || null;
    if (n0) {
      const day = n0.date ? ` on ${n0.date}` : "";
      parts.push(`Note mentions ${n0.keyword || "symptoms"}${day}: ${n0.text}`);
    }
  }

  if (flags.length) {
    if (role === "patient") {
      const lines = [];
      for (const f of flags) {
        const src = String(f.source || "").trim();
        const label = src === "rx" ? "Medication" : src === "labs" ? "Labs" : "Summary";
        const items = (Array.isArray(f.flags) ? f.flags : []).map(plainFlag).filter(Boolean);
        if (items.length) lines.push(`${label}: ${items.join(" · ")}`);
      }
      if (lines.length) parts.push(lines.join("\n"));
    } else {
      const f = flags.map((x) => `${x.source}: ${(x.flags || []).join(" · ")}`).filter(Boolean).join(" | ");
      if (f) parts.push(`Flags: ${f}`);
    }
  }

  const text = parts.filter(Boolean).join(role === "doctor" ? " | " : "\n");
  if (!text) return "";
  if (role === "patient") return safeText(text, 400);
  return safeText(`${headline} — ${text}`, 520);
}

function parsePath(req) {
  const u = new URL(req.url, "http://localhost");
  return { pathname: u.pathname, searchParams: u.searchParams };
}

async function handle(req, res) {
  cors(res);
  if (req.method === "OPTIONS") return json(res, 204, {});

  const { pathname, searchParams } = parsePath(req);
  const path = pathname.replace(/\/+$/, "");

  if (req.method === "GET" && path === "/api/health") {
    return json(res, 200, { ok: true, ts: new Date().toISOString() });
  }

  let db;
  try {
    db = firestore();
  } catch (e) {
    return json(res, 500, { error: String(e?.message || e || "Firestore init failed") });
  }

  if (req.method === "GET" && path === "/api/patients") {
    try {
      const patients = await listPatients(db);
      return json(res, 200, { patients });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  if (req.method === "GET" && path === "/api/doctor/overview") {
    const pipeline = String(searchParams.get("pipeline") || "123").trim();
    try {
      const rows = [];
      const patients = await listPatients(db);
      for (const pid of patients) {
        const checkins = await loadDailyCheckinsRecent(db, pid, 30);
        const lastDt = checkins.length ? String(checkins[checkins.length - 1].date) : "";
        const recent = checkins;
        const lastSum = await loadLatestSummary(db, pid, pipeline);
        const rx = lastSum.prescriptionSummary && typeof lastSum.prescriptionSummary === "object" ? lastSum.prescriptionSummary : {};
        const labs = lastSum.labReportSummary && typeof lastSum.labReportSummary === "object" ? lastSum.labReportSummary : {};
        const alerts = riskAlerts(recent, rx, labs);
        const top = alerts[0] || {};
        rows.push({
          patient_id: pid,
          last_checkin: lastDt,
          risk: top.headline || "",
          active_meds: Array.isArray(rx.active_medications) ? rx.active_medications.length : 0,
          abnormal_labs: Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs.length : 0
        });
      }
      return json(res, 200, { rows });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  const dashMatch = path.match(/^\/api\/patients\/([^/]+)\/dashboard$/);
  if (req.method === "GET" && dashMatch) {
    const patientId = decodeURIComponent(dashMatch[1]);
    const role = String(searchParams.get("role") || "patient").trim().toLowerCase();
    const pipeline = String(searchParams.get("pipeline") || "123").trim();
    const simplifyNotes = new Set(["1", "true", "yes"]).has(String(searchParams.get("simplify_notes") || "").trim().toLowerCase());

    try {
      let start = String(searchParams.get("start") || "").trim();
      let end = String(searchParams.get("end") || "").trim();
      let filtered;
      if (start && end) {
        filtered = await loadDailyCheckinsRange(db, patientId, start, end);
      } else {
        filtered = await loadDailyCheckinsRecent(db, patientId, 30);
        if (filtered.length) {
          end = filtered[filtered.length - 1].date;
          start = filtered[0].date;
        }
      }
      if (!filtered.length) return json(res, 404, { patient_id: patientId, error: "No DailyCheckIns found." });

      const vitalsStats = statsForPeriod(filtered);
      const baseline = baselineMap(filtered, ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"]);

      const latest = await loadLatestSummary(db, patientId, pipeline);
      const rx = latest.prescriptionSummary && typeof latest.prescriptionSummary === "object" ? latest.prescriptionSummary : {};
      const labs = latest.labReportSummary && typeof latest.labReportSummary === "object" ? latest.labReportSummary : {};

      const risk = riskAlerts(filtered, rx, labs);
      const notes = notesFromRows(filtered, role === "patient" ? 3 : 6);

      if (role === "doctor" && simplifyNotes && String(process.env.GOOGLE_API_KEY || "").trim()) {
        const simple = await aiSimpleNotes(notes);
        for (const n of notes) {
          const key = String(n.date || "").slice(0, 10);
          if (simple[key]) n.text = simple[key];
        }
      }

      const uniqueDates = Array.from(new Set(filtered.map((x) => String(x.date || "")).filter(Boolean))).sort();
      const notesMeta = { days: uniqueDates.length, start: uniqueDates[0] || start, end: uniqueDates[uniqueDates.length - 1] || end };

      const active = Array.isArray(rx.active_medications) ? rx.active_medications : [];
      const inactive = Array.isArray(rx.inactive_medications) ? rx.inactive_medications : [];
      const abn = Array.isArray(labs.abnormal_labs) ? labs.abnormal_labs : [];
      const resolved = Array.isArray(labs.resolved_labs) ? labs.resolved_labs : [];

      return json(res, 200, {
        patient_id: patientId,
        start_date: start,
        end_date: end,
        metrics: METRICS,
        baseline,
        checkins: filtered,
        vitals_stats: vitalsStats,
        risk_alerts: risk,
        notes_meta: notesMeta,
        notes,
        rx_glance: {
          active_count: active.length,
          inactive_count: inactive.length,
          risk_flags: Array.isArray(rx.risk_flags) ? rx.risk_flags : []
        },
        labs_glance: {
          abnormal_count: abn.length,
          resolved_count: resolved.length,
          risk_flags: Array.isArray(labs.risk_flags) ? labs.risk_flags : []
        },
        rx: {
          as_of_date: isoDate10(rx.as_of_date || latest.date || end),
          active_medications: normalizeStringList(active, 60),
          inactive_medications: normalizeStringList(inactive, 60),
          risk_flags: normalizeStringList(rx.risk_flags, 20)
        },
        labs: {
          as_of_date: isoDate10(labs.as_of_date || latest.date || end),
          abnormal_labs: normalizeStringList(abn, 60),
          resolved_labs: normalizeStringList(resolved, 60),
          risk_flags: normalizeStringList(labs.risk_flags, 20)
        }
      });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  const tasksMatch = path.match(/^\/api\/patients\/([^/]+)\/action-tasks$/);
  if (req.method === "GET" && tasksMatch) {
    const patientId = decodeURIComponent(tasksMatch[1]);
    const role = String(searchParams.get("role") || "patient").trim().toLowerCase();
    const pipeline = String(searchParams.get("pipeline") || "123").trim();
    try {
      let start = String(searchParams.get("start") || "").trim();
      let end = String(searchParams.get("end") || "").trim();
      let filtered;
      if (start && end) {
        filtered = await loadDailyCheckinsRange(db, patientId, start, end);
      } else {
        filtered = await loadDailyCheckinsRecent(db, patientId, 30);
        if (filtered.length) {
          end = filtered[filtered.length - 1].date;
          start = filtered[0].date;
        }
      }
      if (!filtered.length) return json(res, 404, { patient_id: patientId, error: "No DailyCheckIns found." });

      const vitalsStats = statsForPeriod(filtered);
      const latest = await loadLatestSummary(db, patientId, pipeline);
      const rx = latest.prescriptionSummary && typeof latest.prescriptionSummary === "object" ? latest.prescriptionSummary : {};
      const labs = latest.labReportSummary && typeof latest.labReportSummary === "object" ? latest.labReportSummary : {};
      const notes = notesFromRows(filtered, 12);

      const out = await aiActionTasks(role, patientId, start, end, vitalsStats, notes, rx, labs);
      return json(res, 200, { patient_id: patientId, start_date: start, end_date: end, tasks: out.tasks || [], warning: out.warning || "" });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  const diagMatch = path.match(/^\/api\/patients\/([^/]+)\/diagnosis-alerts$/);
  if (req.method === "GET" && diagMatch) {
    const patientId = decodeURIComponent(diagMatch[1]);
    const role = String(searchParams.get("role") || "patient").trim().toLowerCase();
    const pipeline = String(searchParams.get("pipeline") || "123").trim();
    try {
      let start = String(searchParams.get("start") || "").trim();
      let end = String(searchParams.get("end") || "").trim();
      let filtered;
      if (start && end) {
        filtered = await loadDailyCheckinsRange(db, patientId, start, end);
      } else {
        filtered = await loadDailyCheckinsRecent(db, patientId, 30);
        if (filtered.length) {
          end = filtered[filtered.length - 1].date;
          start = filtered[0].date;
        }
      }
      if (!filtered.length) return json(res, 404, { patient_id: patientId, error: "No DailyCheckIns found." });

      const latest = await loadLatestSummary(db, patientId, pipeline);
      const rx = latest.prescriptionSummary && typeof latest.prescriptionSummary === "object" ? latest.prescriptionSummary : {};
      const labs = latest.labReportSummary && typeof latest.labReportSummary === "object" ? latest.labReportSummary : {};
      const notes = notesFromRows(filtered, 12);

      const out = await aiRiskAlerts(role, patientId, start, end, filtered, notes, rx, labs);
      return json(res, 200, {
        patient_id: patientId,
        start_date: start,
        end_date: end,
        alerts: out?.alerts || [],
        warning: out?.warning || ""
      });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  const aiMatch = path.match(/^\/api\/patients\/([^/]+)\/ai-summary$/);
  if (req.method === "POST" && aiMatch) {
    const patientId = decodeURIComponent(aiMatch[1]);
    try {
      const payload = await readJson(req);
      const role = String(payload.role || "patient").trim().toLowerCase();
      const pipeline = String(payload.pipeline || "123").trim();
      let start = String(payload.start_date || "").trim();
      let end = String(payload.end_date || "").trim();

      let filtered;
      if (start && end) {
        filtered = await loadDailyCheckinsRange(db, patientId, start, end);
      } else {
        filtered = await loadDailyCheckinsRecent(db, patientId, 30);
        if (filtered.length) {
          end = filtered[filtered.length - 1].date;
          start = filtered[0].date;
        }
      }
      if (!filtered.length) return json(res, 404, { error: "No DailyCheckIns found." });
      const vitalsStats = statsForPeriod(filtered);

      const latest = await loadLatestSummary(db, patientId, pipeline);
      const rx = latest.prescriptionSummary && typeof latest.prescriptionSummary === "object" ? latest.prescriptionSummary : {};
      const labs = latest.labReportSummary && typeof latest.labReportSummary === "object" ? latest.labReportSummary : {};

      const text = await aiSummary(role, patientId, start, end, vitalsStats, rx, labs);
      return json(res, 200, { text: text || "" });
    } catch (e) {
      return json(res, 500, { error: String(e?.message || e || "Failed") });
    }
  }

  return json(res, 404, { error: "Not found" });
}

module.exports = async (req, res) => {
  try {
    await handle(req, res);
  } catch (e) {
    cors(res);
    return json(res, 500, { error: String(e?.message || e || "Unhandled") });
  }
};
