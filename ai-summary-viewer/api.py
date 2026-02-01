import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from google import genai
from google.cloud.firestore_v1 import FieldFilter

app = Flask(__name__)


METRICS: Dict[str, Dict[str, str]] = {
    "SBP": {"label": "Systolic BP", "unit": "mmHg", "meaning": "Blood pressure when the heart beats."},
    "DBP": {"label": "Diastolic BP", "unit": "mmHg", "meaning": "Blood pressure when the heart rests."},
    "HR": {"label": "Heart rate", "unit": "bpm", "meaning": "Heart beats per minute."},
    "RR": {"label": "Respiratory rate", "unit": "breaths/min", "meaning": "Breaths per minute."},
    "SpO2": {"label": "Oxygen saturation", "unit": "%", "meaning": "Oxygen level in blood."},
    "Temp": {"label": "Temperature", "unit": "°C", "meaning": "Body temperature."},
    "Pulse": {"label": "Pulse", "unit": "bpm", "meaning": "Pulse rate (may match HR)."},
    "weightKg": {"label": "Weight", "unit": "kg", "meaning": "Body weight."},
    "BMI": {"label": "BMI", "unit": "", "meaning": "Body mass index (weight vs height)."},
}


def _corsify(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.after_request
def _after_request(resp):
    return _corsify(resp)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_bp(bp: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        s = str(bp or "")
        a, b = s.split("/", 1)
        return float(a.strip()), float(b.strip())
    except Exception:
        return None, None


def _firestore_client() -> firestore.Client:
    if not firebase_admin._apps:
        raw = (os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
        if raw:
            firebase_admin.initialize_app(credentials.Certificate(json.loads(raw)))
        else:
            candidates = [
                os.path.join(_project_root(), "serviceAccountKey.json"),
                os.path.join(os.path.dirname(_project_root()), "serviceAccountKey.json"),
            ]
            sa_path = next((p for p in candidates if os.path.exists(p)), "")
            if sa_path:
                firebase_admin.initialize_app(credentials.Certificate(sa_path))
            else:
                firebase_admin.initialize_app()
    return firestore.client()


def _gemini_client() -> Optional[genai.Client]:
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        return None
    return genai.Client(api_key=key)


def list_patients() -> List[str]:
    db = _firestore_client()
    try:
        docs = db.collection("Patients").stream(retry=None)
    except TypeError:
        docs = db.collection("Patients").stream()
    except Exception:
        return []
    out = [d.id for d in docs]
    out.sort()
    return out


def load_daily_checkins(patient_id: str) -> List[Dict[str, Any]]:
    db = _firestore_client()
    col = db.collection("Patients").document(patient_id).collection("DailyCheckIns")
    out: List[Dict[str, Any]] = []
    try:
        try:
            docs = col.stream(retry=None)
        except TypeError:
            docs = col.stream()
        for doc in docs:
            d = doc.to_dict() or {}
            dt = str(d.get("date") or d.get("Date") or d.get("createdAt") or "")[:10]
            vitals = (d.get("vitals") or [])
            v0 = vitals[0] if isinstance(vitals, list) and vitals and isinstance(vitals[0], dict) else {}
            sbp, dbp = _parse_bp(v0.get("BP"))
            out.append(
                {
                    "date": dt,
                    "SBP": sbp,
                    "DBP": dbp,
                    "HR": _parse_float(v0.get("HR")),
                    "RR": _parse_float(v0.get("RR")),
                    "SpO2": _parse_float(v0.get("SpO2")),
                    "Temp": _parse_float(v0.get("Temp")),
                    "Pulse": _parse_float(v0.get("Pulse")),
                    "weightKg": _parse_float(v0.get("weightKg")),
                    "BMI": _parse_float(v0.get("BMI")),
                    "notes_short": str(((d.get("Notes") or {}) if isinstance(d.get("Notes"), dict) else {}).get("short") or ""),
                    "notes_full": str(((d.get("Notes") or {}) if isinstance(d.get("Notes"), dict) else {}).get("full") or ""),
                }
            )
    except Exception:
        return []

    out = [x for x in out if x.get("date")]
    out.sort(key=lambda x: x["date"])
    return out


def load_summaries(patient_id: str, pipeline: str = "123") -> List[Dict[str, Any]]:
    db = _firestore_client()
    q = (
        db.collection("Summaries")
        .where(filter=FieldFilter("patientId", "==", patient_id))
        .where(filter=FieldFilter("pipeline", "==", pipeline))
    )
    out: List[Dict[str, Any]] = []
    try:
        try:
            stream = q.stream(retry=None)
        except TypeError:
            stream = q.stream()
        for s in stream:
            d = s.to_dict() or {}
            dt = str(d.get("date") or "")[:10]
            if not dt:
                continue
            out.append({**d, "_doc_id": s.id})
    except Exception:
        return []
    out.sort(key=lambda x: x.get("date") or "")
    return out


def _latest_summary_on_or_before(summaries: List[Dict[str, Any]], end_date: str) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    for s in summaries:
        dt = str(s.get("date") or "")
        if dt and dt <= end_date:
            best = s
        elif dt and dt > end_date:
            break
    return best


def _filter_range(items: List[Dict[str, Any]], start: str, end: str) -> List[Dict[str, Any]]:
    return [x for x in items if start <= str(x.get("date") or "") <= end]


def _series(values: List[Optional[float]]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float))]


def _stats_for_period(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    metrics = ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"]
    out: Dict[str, Any] = {"days": len(rows), "metrics": {}}
    for m in metrics:
        vals = _series([r.get(m) for r in rows])
        if not vals:
            continue
        out["metrics"][m] = {"min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals), "last": vals[-1]}
    return out


def _baseline_map(rows: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
    if not rows:
        return {}
    base: Dict[str, float] = {}
    first = rows[0]
    for m in metrics:
        v = first.get(m)
        if isinstance(v, (int, float)):
            base[m] = float(v)
    return base


def _risk_topic(flag: str) -> Optional[str]:
    f = " ".join(str(flag or "").strip().lower().split())
    if not f:
        return None
    if any(x in f for x in ["stroke", "tia"]):
        return "stroke"
    if any(x in f for x in ["heart attack", "mi", "myocard"]):
        return "heart attack"
    if any(x in f for x in ["bleed", "hemorr", "haemorr"]):
        return "bleeding"
    if "sepsis" in f:
        return "sepsis"
    if any(x in f for x in ["kidney", "renal", "aki"]):
        return "kidney injury"
    if any(x in f for x in ["hyperkal", "potassium high"]):
        return "high potassium"
    if any(x in f for x in ["hypogly", "glucose low", "low sugar"]):
        return "low blood sugar"
    if any(x in f for x in ["fall", "faint", "syncope"]):
        return "fainting"
    return None


def _risk_alerts(rows: List[Dict[str, Any]], rx: Dict[str, Any], labs: Dict[str, Any]) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    metrics = _stats_for_period(rows).get("metrics", {}) if rows else {}

    sbp_max = metrics.get("SBP", {}).get("max")
    dbp_max = metrics.get("DBP", {}).get("max")
    spo2_min = metrics.get("SpO2", {}).get("min")
    temp_max = metrics.get("Temp", {}).get("max")
    hr_max = metrics.get("HR", {}).get("max")
    hr_min = metrics.get("HR", {}).get("min")

    if sbp_max is not None and dbp_max is not None and (sbp_max >= 180 or dbp_max >= 120):
        alerts.append(
            {
                "severity": "critical",
                "headline": "High risk of hypertensive crisis",
                "action": "Talk to a doctor immediately (or seek urgent care).",
            }
        )
    elif sbp_max is not None and dbp_max is not None and (sbp_max >= 160 or dbp_max >= 100):
        alerts.append({"severity": "high", "headline": "High blood pressure", "action": "Talk to your doctor soon."})

    if spo2_min is not None and spo2_min < 90:
        alerts.append(
            {
                "severity": "critical",
                "headline": "High risk from low oxygen",
                "action": "Talk to a doctor immediately (or seek urgent care).",
            }
        )
    elif spo2_min is not None and spo2_min < 92:
        alerts.append({"severity": "high", "headline": "Low oxygen", "action": "Talk to your doctor soon."})

    if temp_max is not None and temp_max >= 39.0:
        alerts.append({"severity": "high", "headline": "High fever", "action": "Talk to your doctor soon."})
    if hr_max is not None and hr_max >= 130:
        alerts.append({"severity": "high", "headline": "Very high heart rate", "action": "Talk to your doctor soon."})
    if hr_min is not None and hr_min <= 40:
        alerts.append({"severity": "high", "headline": "Very low heart rate", "action": "Talk to your doctor soon."})

    flags: List[str] = []
    flags.extend((rx.get("risk_flags") or []) if isinstance(rx.get("risk_flags"), list) else [])
    flags.extend((labs.get("risk_flags") or []) if isinstance(labs.get("risk_flags"), list) else [])
    for f in flags:
        topic = _risk_topic(str(f))
        if topic:
            alerts.append({"severity": "high", "headline": f"High risk of {topic}", "action": "Talk to your doctor soon."})
        else:
            cleaned = " ".join(str(f or "").strip().split())
            if cleaned:
                alerts.append({"severity": "medium", "headline": f"Risk flag: {cleaned}", "action": "Discuss with your doctor."})

    abn = (labs.get("abnormal_labs") or []) if isinstance(labs.get("abnormal_labs"), list) else []
    if abn:
        names: List[str] = []
        for x in abn[:6]:
            if isinstance(x, str):
                names.append(x)
            elif isinstance(x, dict):
                n = x.get("name") or x.get("test") or x.get("lab") or ""
                if n:
                    names.append(str(n))
        msg = ("Abnormal labs: " + ", ".join(names[:3]) + ".") if names else "Abnormal lab results detected."
        alerts.append({"severity": "medium", "headline": msg, "action": "Discuss with your doctor."})

    order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    seen = set()
    uniq: List[Dict[str, str]] = []
    for a in alerts:
        key = (a.get("severity", ""), a.get("headline", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    uniq.sort(key=lambda x: order.get(x.get("severity") or "", 0), reverse=True)
    return uniq[:5]


def _extract_keywords(text: str) -> List[str]:
    t = " ".join(str(text or "").strip().lower().split())
    if not t:
        return []
    tokens = re.findall(r"[a-z0-9%/]+", t)
    stop = {
        "patient",
        "pt",
        "w",
        "with",
        "and",
        "the",
        "is",
        "are",
        "to",
        "of",
        "bp",
        "hr",
        "rr",
        "spo2",
        "temp",
        "stable",
        "continue",
        "current",
        "management",
        "plan",
        "monitor",
    }
    out = []
    for tok in tokens:
        if tok in stop:
            continue
        if len(tok) < 3:
            continue
        out.append(tok)
    seen = set()
    uniq = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
        if len(uniq) >= 6:
            break
    return uniq


def _note_glance(text: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    parts: List[str] = []
    m = re.search(r"\bBP[:\s]*([0-9]{2,3})\s*/\s*([0-9]{2,3})\b", t, flags=re.IGNORECASE)
    if m:
        parts.append(f"BP {m.group(1)}/{m.group(2)}")
    m = re.search(r"\bHR[:\s]*([0-9]{2,3})\b", t, flags=re.IGNORECASE)
    if m:
        parts.append(f"HR {m.group(1)}")
    m = re.search(r"\bRR[:\s]*([0-9]{1,3})\b", t, flags=re.IGNORECASE)
    if m:
        parts.append(f"RR {m.group(1)}")
    m = re.search(r"\bSpO2[:\s]*([0-9]{2,3})\s*%?\b", t, flags=re.IGNORECASE)
    if m:
        parts.append(f"SpO2 {m.group(1)}%")
    m = re.search(r"\bTemp(?:erature)?[:\s]*([0-9]{2,3}(?:\.[0-9])?)\b", t, flags=re.IGNORECASE)
    if m:
        parts.append(f"Temp {m.group(1)}°C")
    return " • ".join(parts)


def _notes(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in reversed(rows):
        s = (row.get("notes_short") or "").strip() or (row.get("notes_full") or "").strip()
        if not s:
            continue
        body = " ".join(s.split())
        out.append(
            {
                "date": row.get("date") or "",
                "glance": _note_glance(body),
                "keywords": _extract_keywords(body),
                "text": (body[:117].rstrip() + "...") if len(body) > 120 else body,
                "raw": body,
            }
        )
        if len(out) >= limit:
            break
    return out


def _ai_simple_notes(notes: List[Dict[str, Any]]) -> Dict[str, str]:
    client = _gemini_client()
    if client is None:
        return {}
    items = [{"date": str(n.get("date") or ""), "note": str(n.get("raw") or "")[:500]} for n in notes]
    prompt = (
        "Rewrite each clinical note in simple English for quick reading.\n"
        "Rules:\n"
        "- One short sentence per note (max 14 words)\n"
        "- Keep key vitals if present (BP/HR/SpO2/Temp)\n"
        "- Keep the main issue (e.g., BP elevated, stable, improving)\n"
        "- Output ONLY valid JSON: {\"YYYY-MM-DD\": \"sentence\", ...}\n\n"
        f"JSON:\n{json.dumps({'notes': items}, indent=2)}"
    )
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = str(getattr(resp, "text", "") or "").strip()
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
    except Exception:
        return {}
    if not isinstance(out, dict):
        return {}
    cleaned: Dict[str, str] = {}
    for k, v in out.items():
        kk = str(k or "").strip()[:10]
        vv = " ".join(str(v or "").strip().split())
        if kk and vv:
            cleaned[kk] = vv
    return cleaned


def _ai_summary(role: str, patient_id: str, start_date: str, end_date: str, vitals_stats: Dict[str, Any], rx: Dict[str, Any], labs: Dict[str, Any]) -> str:
    client = _gemini_client()
    if client is None:
        return ""
    payload = {
        "role": role,
        "patient_id": patient_id,
        "start_date": start_date,
        "end_date": end_date,
        "vitals_stats": vitals_stats,
        "prescriptions_asof": rx,
        "labs_asof": labs,
    }
    if role == "patient":
        prompt = (
            "You are a patient-friendly health dashboard assistant.\n"
            "Summarize the selected period in simple language.\n"
            "Focus on: (1) what changed, (2) what to watch, (3) questions to ask the doctor.\n"
            "Keep it short and easy to scan. Use bullet points.\n\n"
            f"JSON:\n{json.dumps(payload, indent=2, default=str)}"
        )
    else:
        prompt = (
            "You are a clinician-facing dashboard assistant.\n"
            "Summarize the selected period for rapid review.\n"
            "Focus on: changes vs baseline, potential concerns, medication/lab deltas, and next actions/questions.\n"
            "Use concise bullet points.\n\n"
            f"JSON:\n{json.dumps(payload, indent=2, default=str)}"
        )
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return str(getattr(resp, "text", "") or "").strip()


@app.route("/api/patients", methods=["GET", "OPTIONS"])
def api_patients():
    if request.method == "OPTIONS":
        return _corsify(app.make_response(("", 204)))
    return jsonify({"patients": list_patients()})


@app.route("/api/patients/<patient_id>/dashboard", methods=["GET", "OPTIONS"])
def api_patient_dashboard(patient_id: str):
    if request.method == "OPTIONS":
        return _corsify(app.make_response(("", 204)))

    start = (request.args.get("start") or "").strip()
    end = (request.args.get("end") or "").strip()
    pipeline = (request.args.get("pipeline") or "123").strip()
    role = (request.args.get("role") or "patient").strip().lower()
    simplify_notes = (request.args.get("simplify_notes") or "").strip() in {"1", "true", "yes"}

    checkins = load_daily_checkins(patient_id)
    if not checkins:
        return jsonify({"patient_id": patient_id, "error": "No DailyCheckIns found."}), 404

    if not start or not end:
        end = checkins[-1]["date"]
        start = checkins[max(0, len(checkins) - 30)]["date"]

    filtered = _filter_range(checkins, start=start, end=end)
    vitals_stats = _stats_for_period(filtered)
    baseline = _baseline_map(filtered, ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"])

    summaries = load_summaries(patient_id, pipeline=pipeline)
    latest = _latest_summary_on_or_before(summaries, end_date=end)
    rx = latest.get("prescriptionSummary") if isinstance(latest.get("prescriptionSummary"), dict) else {}
    labs = latest.get("labReportSummary") if isinstance(latest.get("labReportSummary"), dict) else {}

    risk = _risk_alerts(filtered, rx, labs)
    notes = _notes(filtered, limit=3 if role == "patient" else 6)
    if simplify_notes and role == "doctor":
        simple = _ai_simple_notes(notes)
        for n in notes:
            s = simple.get(str(n.get("date") or "")[:10], "")
            if s:
                n["text"] = s

    dates = sorted({str(x.get("date") or "") for x in filtered if x.get("date")})
    notes_meta = {"days": len(dates), "start": dates[0] if dates else start, "end": dates[-1] if dates else end}

    active = (rx.get("active_medications") or []) if isinstance(rx.get("active_medications"), list) else []
    inactive = (rx.get("inactive_medications") or []) if isinstance(rx.get("inactive_medications"), list) else []
    abn = (labs.get("abnormal_labs") or []) if isinstance(labs.get("abnormal_labs"), list) else []
    res = (labs.get("resolved_labs") or []) if isinstance(labs.get("resolved_labs"), list) else []

    return jsonify(
        {
            "patient_id": patient_id,
            "start_date": start,
            "end_date": end,
            "metrics": METRICS,
            "baseline": baseline,
            "checkins": filtered,
            "vitals_stats": vitals_stats,
            "risk_alerts": risk,
            "notes_meta": notes_meta,
            "notes": notes,
            "rx_glance": {
                "active_count": len(active),
                "inactive_count": len(inactive),
                "risk_flags": (rx.get("risk_flags") or []) if isinstance(rx.get("risk_flags"), list) else [],
            },
            "labs_glance": {
                "abnormal_count": len(abn),
                "resolved_count": len(res),
                "risk_flags": (labs.get("risk_flags") or []) if isinstance(labs.get("risk_flags"), list) else [],
            },
        }
    )


@app.route("/api/doctor/overview", methods=["GET", "OPTIONS"])
def api_doctor_overview():
    if request.method == "OPTIONS":
        return _corsify(app.make_response(("", 204)))

    pipeline = (request.args.get("pipeline") or "123").strip()
    out = []
    for pid in list_patients():
        checkins = load_daily_checkins(pid)
        last_dt = str(checkins[-1]["date"]) if checkins else ""
        recent = checkins[-30:] if len(checkins) > 30 else checkins
        summaries = load_summaries(pid, pipeline=pipeline)
        last_sum = summaries[-1] if summaries else {}
        rx = last_sum.get("prescriptionSummary") if isinstance(last_sum.get("prescriptionSummary"), dict) else {}
        labs = last_sum.get("labReportSummary") if isinstance(last_sum.get("labReportSummary"), dict) else {}
        alerts = _risk_alerts(recent, rx, labs)
        top = alerts[0] if alerts else {}
        out.append(
            {
                "patient_id": pid,
                "last_checkin": last_dt,
                "risk": top.get("headline") or "",
                "active_meds": len((rx.get("active_medications") or []) if isinstance(rx.get("active_medications"), list) else []),
                "abnormal_labs": len((labs.get("abnormal_labs") or []) if isinstance(labs.get("abnormal_labs"), list) else []),
            }
        )
    return jsonify({"rows": out})


@app.route("/api/patients/<patient_id>/ai-summary", methods=["POST", "OPTIONS"])
def api_patient_ai_summary(patient_id: str):
    if request.method == "OPTIONS":
        return _corsify(app.make_response(("", 204)))

    payload = request.get_json(silent=True) or {}
    role = str(payload.get("role") or "patient").strip().lower()
    start = str(payload.get("start_date") or "").strip()
    end = str(payload.get("end_date") or "").strip()
    pipeline = str(payload.get("pipeline") or "123").strip()

    checkins = load_daily_checkins(patient_id)
    if not checkins:
        return jsonify({"error": "No DailyCheckIns found."}), 404
    if not start or not end:
        end = checkins[-1]["date"]
        start = checkins[max(0, len(checkins) - 30)]["date"]
    filtered = _filter_range(checkins, start=start, end=end)
    vitals_stats = _stats_for_period(filtered)

    summaries = load_summaries(patient_id, pipeline=pipeline)
    latest = _latest_summary_on_or_before(summaries, end_date=end)
    rx = latest.get("prescriptionSummary") if isinstance(latest.get("prescriptionSummary"), dict) else {}
    labs = latest.get("labReportSummary") if isinstance(latest.get("labReportSummary"), dict) else {}

    text = _ai_summary(role, patient_id, start, end, vitals_stats, rx, labs)
    return jsonify({"text": text})


if __name__ == "__main__":
    port = int(os.getenv("PORT") or "8080")
    app.run(host="0.0.0.0", port=port)
