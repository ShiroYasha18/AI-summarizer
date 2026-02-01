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
  if (rx && Array.isArray(rx.risk_flags)) flags.push(...rx.risk_flags);
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

async function loadDailyCheckins(db, patientId) {
  const snap = await db.collection("Patients").doc(patientId).collection("DailyCheckIns").get();
  const out = [];
  for (const doc of snap.docs) {
    const d = doc.data() || {};
    const dt = isoDate10(d.date || d.Date || d.createdAt || "");
    const vitals = Array.isArray(d.vitals) ? d.vitals : [];
    const v0 = vitals.length && vitals[0] && typeof vitals[0] === "object" ? vitals[0] : {};
    const [sbp, dbp] = parseBp(v0.BP);
    const notesObj = d.Notes && typeof d.Notes === "object" ? d.Notes : {};
    out.push({
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
    });
  }
  const cleaned = out.filter((x) => x.date);
  cleaned.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  return cleaned;
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
        const checkins = await loadDailyCheckins(db, pid);
        const lastDt = checkins.length ? String(checkins[checkins.length - 1].date) : "";
        const recent = checkins.length > 30 ? checkins.slice(-30) : checkins;
        const summaries = await loadSummaries(db, pid, pipeline);
        const lastSum = summaries.length ? summaries[summaries.length - 1] : {};
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
      const checkins = await loadDailyCheckins(db, patientId);
      if (!checkins.length) return json(res, 404, { patient_id: patientId, error: "No DailyCheckIns found." });

      let start = String(searchParams.get("start") || "").trim();
      let end = String(searchParams.get("end") || "").trim();
      if (!start || !end) {
        end = checkins[checkins.length - 1].date;
        start = checkins[Math.max(0, checkins.length - 30)].date;
      }

      const filtered = filterRange(checkins, start, end);
      const vitalsStats = statsForPeriod(filtered);
      const baseline = baselineMap(filtered, ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"]);

      const summaries = await loadSummaries(db, patientId, pipeline);
      const latest = latestSummaryOnOrBefore(summaries, end);
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

  const aiMatch = path.match(/^\/api\/patients\/([^/]+)\/ai-summary$/);
  if (req.method === "POST" && aiMatch) {
    const patientId = decodeURIComponent(aiMatch[1]);
    try {
      const payload = await readJson(req);
      const role = String(payload.role || "patient").trim().toLowerCase();
      const pipeline = String(payload.pipeline || "123").trim();
      let start = String(payload.start_date || "").trim();
      let end = String(payload.end_date || "").trim();

      const checkins = await loadDailyCheckins(db, patientId);
      if (!checkins.length) return json(res, 404, { error: "No DailyCheckIns found." });
      if (!start || !end) {
        end = checkins[checkins.length - 1].date;
        start = checkins[Math.max(0, checkins.length - 30)].date;
      }
      const filtered = filterRange(checkins, start, end);
      const vitalsStats = statsForPeriod(filtered);

      const summaries = await loadSummaries(db, patientId, pipeline);
      const latest = latestSummaryOnOrBefore(summaries, end);
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
