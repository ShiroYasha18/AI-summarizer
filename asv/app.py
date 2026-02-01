import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, firestore
from google import genai
from google.cloud.firestore_v1 import FieldFilter


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


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    import sys

    root = _project_root()
    sys.path.append(str(root))
    from main import load_env_file

    load_env_file(root / ".env")


@st.cache_resource
def _firestore_client() -> firestore.Client:
    root = _project_root()
    sa = root / "serviceAccountKey.json"
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(sa)))
    return firestore.client()


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


@st.cache_data(ttl=30)
def list_patients() -> List[str]:
    db = _firestore_client()
    try:
        docs = db.collection("Patients").stream(retry=None)
    except TypeError:
        docs = db.collection("Patients").stream()
    except Exception:
        return []
    ids = [d.id for d in docs]
    ids.sort()
    return ids


@st.cache_data(ttl=30)
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


@st.cache_data(ttl=30)
def load_summaries(patient_id: str, pipeline: str = "123") -> List[Dict[str, Any]]:
    db = _firestore_client()
    docs = (
        db.collection("Summaries")
        .where(filter=FieldFilter("patientId", "==", patient_id))
        .where(filter=FieldFilter("pipeline", "==", pipeline))
    )
    out = []
    try:
        try:
            stream = docs.stream(retry=None)
        except TypeError:
            stream = docs.stream()
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


def _filter_range(items: List[Dict[str, Any]], start: str, end: str) -> List[Dict[str, Any]]:
    return [x for x in items if start <= str(x.get("date") or "") <= end]


def _date_picker(checkins: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not checkins:
        today = datetime.now(timezone.utc).date().isoformat()
        return today, today

    all_dates = [str(x.get("date") or "") for x in checkins]
    max_dt = max(all_dates)
    max_d = datetime.strptime(max_dt, "%Y-%m-%d").date()

    preset = st.selectbox("Range", ["7 days", "10 days", "15 days", "30 days", "Custom"], index=3)
    if preset != "Custom":
        days = int(preset.split()[0])
        start_d = max_d - timedelta(days=max(0, days - 1))
        return start_d.isoformat(), max_d.isoformat()

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start date", value=max_d - timedelta(days=29))
    with c2:
        end = st.date_input("End date", value=max_d)
    if start > end:
        start, end = end, start
    return start.isoformat(), end.isoformat()


def _to_df(checkins: List[Dict[str, Any]]) -> pd.DataFrame:
    if not checkins:
        return pd.DataFrame()
    df = pd.DataFrame(checkins)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df


def _latest_summary_on_or_before(summaries: List[Dict[str, Any]], end_date: str) -> Optional[Dict[str, Any]]:
    best = None
    for s in summaries:
        dt = str(s.get("date") or "")
        if dt and dt <= end_date:
            best = s
        elif dt and dt > end_date:
            break
    return best


def _stats_for_period(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    metrics = ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"]
    out = {"days": int(len(df)), "metrics": {}}
    for m in metrics:
        if m not in df.columns:
            continue
        s = pd.to_numeric(df[m], errors="coerce")
        out["metrics"][m] = {
            "min": None if s.dropna().empty else float(s.min()),
            "max": None if s.dropna().empty else float(s.max()),
            "avg": None if s.dropna().empty else float(s.mean()),
            "last": None if s.dropna().empty else float(s.dropna().iloc[-1]),
        }
    return out


@st.cache_resource
def _gemini_client() -> Optional[genai.Client]:
    _load_env()
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        return None
    return genai.Client(api_key=key)


def _metric_value(df: pd.DataFrame, col: str, which: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    if which == "first":
        return float(s.iloc[0])
    return float(s.iloc[-1])


def _baseline_map(df: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in metrics:
        v = _metric_value(df, m, "first")
        if v is None:
            continue
        out[m] = v
    return out


def _fmt(v: Optional[float], decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def _delta_str(a: Optional[float], b: Optional[float], decimals: int = 1) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.{decimals}f}"


def _series_min(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min())


def _series_max(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.max())


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


def _risk_alerts(df: pd.DataFrame, rx: Dict[str, Any], labs: Dict[str, Any]) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []

    sbp_max = _series_max(df, "SBP")
    dbp_max = _series_max(df, "DBP")
    spo2_min = _series_min(df, "SpO2")
    temp_max = _series_max(df, "Temp")
    hr_max = _series_max(df, "HR")
    hr_min = _series_min(df, "HR")

    if sbp_max is not None and dbp_max is not None and (sbp_max >= 180 or dbp_max >= 120):
        alerts.append(
            {
                "severity": "critical",
                "headline": "High risk of hypertensive crisis",
                "action": "Talk to a doctor immediately (or seek urgent care).",
            }
        )
    elif sbp_max is not None and dbp_max is not None and (sbp_max >= 160 or dbp_max >= 100):
        alerts.append(
            {
                "severity": "high",
                "headline": "High blood pressure",
                "action": "Talk to your doctor soon.",
            }
        )

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
            alerts.append(
                {
                    "severity": "high",
                    "headline": f"High risk of {topic}",
                    "action": "Talk to your doctor soon.",
                }
            )
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
    uniq = []
    for a in alerts:
        key = (a.get("severity", ""), a.get("headline", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    uniq.sort(key=lambda x: order.get(x.get("severity") or "", 0), reverse=True)
    return uniq[:5]


def _render_risk_alerts(alerts: List[Dict[str, str]]) -> None:
    if not alerts:
        st.success("No urgent risk flags detected in this period.")
        return

    st.subheader("Risk Alerts")
    for a in alerts:
        sev = a.get("severity") or "medium"
        headline = a.get("headline") or ""
        action = a.get("action") or ""
        text = f"{headline}\n\n{action}".strip()
        if sev == "critical":
            st.error(text)
        elif sev == "high":
            st.warning(text)
        else:
            st.info(text)


def _metric_help(role: str) -> None:
    if role == "patient":
        with st.expander("What do these mean?"):
            rows = []
            for k, meta in METRICS.items():
                rows.append({"metric": k, "name": meta["label"], "unit": meta["unit"], "meaning": meta["meaning"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        with st.expander("Metric definitions"):
            for k, meta in METRICS.items():
                unit = f" ({meta['unit']})" if meta["unit"] else ""
                st.write(f"- {k}: {meta['label']}{unit} — {meta['meaning']}")


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

    if parts:
        return " • ".join(parts)
    return ""


def _note_cards(checkins: List[Dict[str, Any]], role: str) -> None:
    notes = []
    for row in reversed(checkins):
        s = (row.get("notes_short") or "").strip() or (row.get("notes_full") or "").strip()
        if not s:
            continue
        notes.append({"date": row["date"], "note": s})
        if len(notes) >= 6:
            break
    if not notes:
        st.caption("No notes in this range.")
        return

    st.subheader("Notes (At A Glance)")
    dates = sorted({str(x.get("date") or "") for x in checkins if x.get("date")})
    days = len(dates)
    if dates:
        st.caption(f"Showing notes from last {days} days ({dates[0]} → {dates[-1]}).")

    show = notes[:3] if role == "patient" else notes[:6]
    ai_simple = False
    simplified: Dict[str, str] = {}
    if role == "doctor":
        ai_simple = st.checkbox("Simplify notes (AI)", value=False)
        if ai_simple:
            simplified = _simplify_notes_with_ai(show)

    cols = st.columns(1 if role == "patient" else 2)
    for i, n in enumerate(show):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**{n['date']}**")
                glance = _note_glance(n["note"])
                if glance:
                    st.caption(glance)
                kws = _extract_keywords(n["note"])
                if kws:
                    st.caption("Keywords: " + " · ".join(kws))
                body = simplified.get(n["date"], "") if ai_simple else ""
                if not body:
                    body = " ".join(str(n["note"] or "").strip().split())
                if len(body) > 120:
                    body = body[:117].rstrip() + "..."
                st.write(body)

    if role == "doctor" and st.checkbox("Show full notes table"):
        st.dataframe(pd.DataFrame(list(reversed(notes))), use_container_width=True, hide_index=True)


def _ai_insights(
    role: str,
    patient_id: str,
    start_date: str,
    end_date: str,
    vitals_stats: Dict[str, Any],
    rx: Dict[str, Any],
    labs: Dict[str, Any],
) -> str:
    client = _gemini_client()
    if client is None:
        return "GOOGLE_API_KEY is not set."

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


@st.cache_data(ttl=300)
def _simplify_notes_with_ai(notes: List[Dict[str, str]]) -> Dict[str, str]:
    client = _gemini_client()
    if client is None:
        return {}

    items = [{"date": str(n.get("date") or ""), "note": str(n.get("note") or "")[:500]} for n in notes]
    payload = {"notes": items}
    prompt = (
        "Rewrite each clinical note in simple English for quick reading.\n"
        "Rules:\n"
        "- One short sentence per note (max 14 words)\n"
        "- Keep key vitals if present (BP/HR/SpO2/Temp)\n"
        "- Keep the main issue (e.g., BP elevated, stable, improving)\n"
        "- Output ONLY valid JSON: {\"YYYY-MM-DD\": \"sentence\", ...}\n\n"
        f"JSON:\n{json.dumps(payload, indent=2)}"
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
        if isinstance(out, dict):
            cleaned: Dict[str, str] = {}
            for k, v in out.items():
                kk = str(k or "").strip()[:10]
                vv = " ".join(str(v or "").strip().split())
                if kk and vv:
                    cleaned[kk] = vv
            return cleaned
    except Exception:
        return {}
    return {}


def _vitals_overview(df: pd.DataFrame, role: str) -> Dict[str, Any]:
    if df.empty:
        return {}

    sbp_last = _metric_value(df, "SBP", "last")
    dbp_last = _metric_value(df, "DBP", "last")
    sbp_first = _metric_value(df, "SBP", "first")
    dbp_first = _metric_value(df, "DBP", "first")

    hr_last = _metric_value(df, "HR", "last")
    hr_first = _metric_value(df, "HR", "first")

    spo2_last = _metric_value(df, "SpO2", "last")
    spo2_first = _metric_value(df, "SpO2", "first")

    temp_last = _metric_value(df, "Temp", "last")
    temp_first = _metric_value(df, "Temp", "first")

    w_last = _metric_value(df, "weightKg", "last")
    w_first = _metric_value(df, "weightKg", "first")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Blood pressure",
            f"{_fmt(sbp_last, 0)}/{_fmt(dbp_last, 0)} mmHg",
            delta=(
                f"{_delta_str(sbp_first, sbp_last, 0)}/{_delta_str(dbp_first, dbp_last, 0)}"
                if sbp_last is not None and dbp_last is not None and sbp_first is not None and dbp_first is not None
                else None
            ),
        )
    with c2:
        st.metric("Heart rate", f"{_fmt(hr_last, 0)} bpm", delta=_delta_str(hr_first, hr_last, 0) or None)
    with c3:
        st.metric("Oxygen", f"{_fmt(spo2_last, 0)}%", delta=_delta_str(spo2_first, spo2_last, 0) or None)
    with c4:
        st.metric("Temperature", f"{_fmt(temp_last, 1)} °C", delta=_delta_str(temp_first, temp_last, 1) or None)
    with c5:
        st.metric("Weight", f"{_fmt(w_last, 1)} kg", delta=_delta_str(w_first, w_last, 1) or None)

    if role == "patient":
        st.caption("Up/down shows change vs baseline (first day in the selected range).")
    else:
        st.caption("Delta is last value minus baseline (first day in the selected range).")
    return _stats_for_period(df)


def _plot_vitals_patient(df: pd.DataFrame) -> None:
    st.subheader("Trends")
    if df.empty:
        st.info("No data in the selected range.")
        return

    st.caption("Baseline (dashed) = first reading in the selected range.")
    view = st.selectbox("Show", ["Blood pressure", "Heart & oxygen", "Temperature", "Weight & BMI"], index=0)
    dfr = df.reset_index()
    rename = {m: METRICS[m]["label"] for m in ["SBP", "DBP", "HR", "SpO2", "Temp", "weightKg", "BMI"] if m in dfr.columns}
    base = _baseline_map(df, ["SBP", "DBP", "HR", "SpO2", "Temp", "weightKg", "BMI"])

    if view == "Blood pressure":
        picked = [m for m in ["SBP", "DBP"] if m in dfr.columns]
        long = dfr.melt(id_vars=["date"], value_vars=picked, var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        long["metric"] = long["metric"].map(lambda x: rename.get(x, x))
        baseline_rows = [
            {"metric": rename.get("SBP", "SBP"), "baseline": base.get("SBP")}
            if base.get("SBP") is not None
            else None,
            {"metric": rename.get("DBP", "DBP"), "baseline": base.get("DBP")}
            if base.get("DBP") is not None
            else None,
        ]
        baseline_rows = [x for x in baseline_rows if x]
        baseline = (
            alt.Chart(pd.DataFrame(baseline_rows))
            .mark_rule(strokeDash=[6, 4], opacity=0.55)
            .encode(y=alt.Y("baseline:Q"), color=alt.Color("metric:N", legend=None), tooltip=["metric:N", "baseline:Q"])
        )
        chart = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("metric:N", legend=alt.Legend(title=None)),
                tooltip=["date:T", "metric:N", "value:Q"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart + baseline, use_container_width=True)
        return

    if view == "Heart & oxygen":
        picked = [m for m in ["HR", "SpO2"] if m in dfr.columns]
        long = dfr.melt(id_vars=["date"], value_vars=picked, var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        long["metric"] = long["metric"].map(lambda x: rename.get(x, x))
        baseline_rows = [
            {"metric": rename.get("HR", "HR"), "baseline": base.get("HR")} if base.get("HR") is not None else None,
            {"metric": rename.get("SpO2", "SpO2"), "baseline": base.get("SpO2")}
            if base.get("SpO2") is not None
            else None,
        ]
        baseline_rows = [x for x in baseline_rows if x]
        baseline = (
            alt.Chart(pd.DataFrame(baseline_rows))
            .mark_rule(strokeDash=[6, 4], opacity=0.55)
            .encode(y=alt.Y("baseline:Q"), color=alt.Color("metric:N", legend=None), tooltip=["metric:N", "baseline:Q"])
        )
        chart = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("metric:N", legend=alt.Legend(title=None)),
                tooltip=["date:T", "metric:N", "value:Q"],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart + baseline, use_container_width=True)
        return

    if view == "Temperature":
        if "Temp" not in dfr.columns:
            st.info("No temperature data in the selected range.")
            return
        long = dfr[["date", "Temp"]].copy()
        long["Temp"] = pd.to_numeric(long["Temp"], errors="coerce")
        long = long.dropna(subset=["Temp"])
        baseline_val = base.get("Temp")
        baseline = (
            alt.Chart(pd.DataFrame([{"baseline": baseline_val}]))
            .mark_rule(strokeDash=[6, 4], opacity=0.55, color="#9aa0a6")
            .encode(y=alt.Y("baseline:Q"), tooltip=[alt.Tooltip("baseline:Q", title="Baseline")])
            if baseline_val is not None
            else alt.Chart(pd.DataFrame([{"baseline": 0}])).mark_rule(opacity=0)
        )
        chart = (
            alt.Chart(long)
            .mark_area()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("Temp:Q", title=None),
                tooltip=["date:T", alt.Tooltip("Temp:Q", title=rename.get("Temp", "Temp"))],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart + baseline, use_container_width=True)
        return

    c1, c2 = st.columns(2)
    with c1:
        if "weightKg" not in dfr.columns:
            st.info("No weight data in the selected range.")
        else:
            w = dfr[["date", "weightKg"]].copy()
            w["weightKg"] = pd.to_numeric(w["weightKg"], errors="coerce")
            w = w.dropna(subset=["weightKg"])
            baseline_val = base.get("weightKg")
            baseline = (
                alt.Chart(pd.DataFrame([{"baseline": baseline_val}]))
                .mark_rule(strokeDash=[6, 4], opacity=0.55, color="#9aa0a6")
                .encode(y=alt.Y("baseline:Q"), tooltip=[alt.Tooltip("baseline:Q", title="Baseline")])
                if baseline_val is not None
                else alt.Chart(pd.DataFrame([{"baseline": 0}])).mark_rule(opacity=0)
            )
            chart = (
                alt.Chart(w)
                .mark_bar()
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("weightKg:Q", title=None),
                    tooltip=["date:T", alt.Tooltip("weightKg:Q", title=rename.get("weightKg", "weightKg"))],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart + baseline, use_container_width=True)
    with c2:
        if "BMI" not in dfr.columns:
            st.info("No BMI data in the selected range.")
        else:
            b = dfr[["date", "BMI"]].copy()
            b["BMI"] = pd.to_numeric(b["BMI"], errors="coerce")
            b = b.dropna(subset=["BMI"])
            baseline_val = base.get("BMI")
            baseline = (
                alt.Chart(pd.DataFrame([{"baseline": baseline_val}]))
                .mark_rule(strokeDash=[6, 4], opacity=0.55, color="#9aa0a6")
                .encode(y=alt.Y("baseline:Q"), tooltip=[alt.Tooltip("baseline:Q", title="Baseline")])
                if baseline_val is not None
                else alt.Chart(pd.DataFrame([{"baseline": 0}])).mark_rule(opacity=0)
            )
            chart = (
                alt.Chart(b)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("BMI:Q", title=None),
                    tooltip=["date:T", alt.Tooltip("BMI:Q", title=rename.get("BMI", "BMI"))],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart + baseline, use_container_width=True)


def _plot_vitals_doctor(df: pd.DataFrame) -> None:
    st.subheader("Vitals (Interactive)")
    if df.empty:
        st.info("No data in the selected range.")
        return

    available = [m for m in ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"] if m in df.columns]
    picked = st.multiselect("Metrics", available, default=[m for m in ["SBP", "DBP", "HR", "SpO2"] if m in available])
    if not picked:
        return
    dfr = df.reset_index()
    long = dfr.melt(id_vars=["date"], value_vars=picked, var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long["metric_label"] = long["metric"].map(lambda x: f"{x} ({METRICS.get(x, {}).get('unit','')})".strip())
    long["metric_name"] = long["metric"].map(lambda x: METRICS.get(x, {}).get("label", x))

    base = alt.Chart(long).encode(x=alt.X("date:T", title="Date"))
    lines = base.mark_line(point=True).encode(
        y=alt.Y("value:Q", title=None),
        color=alt.Color("metric_name:N", legend=alt.Legend(title="Metric")),
        tooltip=["date:T", "metric:N", "metric_name:N", "value:Q"],
    )
    st.altair_chart(lines.properties(height=340).interactive(), use_container_width=True)

    if st.checkbox("Show distribution (box plot)"):
        box = (
            alt.Chart(long)
            .mark_boxplot()
            .encode(x=alt.X("metric:N", title=None), y=alt.Y("value:Q", title=None), color=alt.Color("metric:N", legend=None))
            .properties(height=220)
        )
        st.altair_chart(box, use_container_width=True)


def render_patient_view(patient_id: str) -> None:
    checkins = load_daily_checkins(patient_id)
    if not checkins:
        st.warning("No DailyCheckIns found.")
        return

    start_date, end_date = _date_picker(checkins)
    filtered = _filter_range(checkins, start=start_date, end=end_date)
    df = _to_df(filtered)

    st.caption(f"Showing {start_date} → {end_date}")
    summaries = load_summaries(patient_id, pipeline="123")
    latest = _latest_summary_on_or_before(summaries, end_date=end_date) or {}
    rx = (latest.get("prescriptionSummary") or {}) if isinstance(latest.get("prescriptionSummary"), dict) else {}
    labs = (latest.get("labReportSummary") or {}) if isinstance(latest.get("labReportSummary"), dict) else {}

    _render_risk_alerts(_risk_alerts(df, rx, labs))
    vitals_stats = _vitals_overview(df, role="patient")
    _metric_help(role="patient")
    _plot_vitals_patient(df)

    _note_cards(filtered, role="patient")

    st.subheader("Medications (At A Glance)")
    active = (rx.get("active_medications") or []) if isinstance(rx.get("active_medications"), list) else []
    inactive = (rx.get("inactive_medications") or []) if isinstance(rx.get("inactive_medications"), list) else []
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Active meds", str(len(active)))
    with c2:
        st.metric("Inactive meds", str(len(inactive)))
    with c3:
        st.metric("Rx risk flags", str(len((rx.get("risk_flags") or []) if isinstance(rx.get("risk_flags"), list) else [])))
    with st.expander("Details"):
        st.json(rx if rx else {"patient_id": patient_id, "as_of_date": end_date, "active_medications": [], "inactive_medications": []})

    st.subheader("Lab Reports (At A Glance)")
    abn = (labs.get("abnormal_labs") or []) if isinstance(labs.get("abnormal_labs"), list) else []
    res = (labs.get("resolved_labs") or []) if isinstance(labs.get("resolved_labs"), list) else []
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Abnormal", str(len(abn)))
    with c2:
        st.metric("Resolved", str(len(res)))
    with c3:
        st.metric("Lab risk flags", str(len((labs.get("risk_flags") or []) if isinstance(labs.get("risk_flags"), list) else [])))
    with st.expander("Details"):
        st.json(labs if labs else {"patient_id": patient_id, "as_of_date": end_date, "abnormal_labs": [], "resolved_labs": []})

    st.subheader("AI Summary")
    if st.button("Generate AI insights", type="primary"):
        text = _ai_insights("patient", patient_id, start_date, end_date, vitals_stats, rx, labs)
        st.write(text)


def render_doctor_patient_view(patient_id: str) -> None:
    checkins = load_daily_checkins(patient_id)
    if not checkins:
        st.warning("No DailyCheckIns found.")
        return

    start_date, end_date = _date_picker(checkins)
    filtered = _filter_range(checkins, start=start_date, end=end_date)
    df = _to_df(filtered)

    st.caption(f"Showing {start_date} → {end_date}")
    summaries = load_summaries(patient_id, pipeline="123")
    latest = _latest_summary_on_or_before(summaries, end_date=end_date) or {}
    rx = (latest.get("prescriptionSummary") or {}) if isinstance(latest.get("prescriptionSummary"), dict) else {}
    labs = (latest.get("labReportSummary") or {}) if isinstance(latest.get("labReportSummary"), dict) else {}

    _render_risk_alerts(_risk_alerts(df, rx, labs))
    vitals_stats = _vitals_overview(df, role="doctor")
    _metric_help(role="doctor")
    _plot_vitals_doctor(df)

    _note_cards(filtered, role="doctor")

    st.subheader("Medications (As Of End Date)")
    st.json(rx if rx else {"patient_id": patient_id, "as_of_date": end_date, "active_medications": [], "inactive_medications": []})

    st.subheader("Lab Reports (As Of End Date)")
    st.json(labs if labs else {"patient_id": patient_id, "as_of_date": end_date, "abnormal_labs": [], "resolved_labs": []})

    st.subheader("AI Summary")
    if st.button("Generate AI insights", type="primary"):
        text = _ai_insights("doctor", patient_id, start_date, end_date, vitals_stats, rx, labs)
        st.write(text)


def render_doctor_view() -> None:
    patients = list_patients()
    rows = []
    for pid in patients:
        checkins = load_daily_checkins(pid)
        last_dt = str(checkins[-1]["date"]) if checkins else ""
        summaries = load_summaries(pid, pipeline="123")
        last_sum = summaries[-1] if summaries else {}
        rx = (last_sum.get("prescriptionSummary") or {}) if isinstance(last_sum.get("prescriptionSummary"), dict) else {}
        labs = (last_sum.get("labReportSummary") or {}) if isinstance(last_sum.get("labReportSummary"), dict) else {}
        df = _to_df(checkins[-30:] if len(checkins) > 30 else checkins)
        alerts = _risk_alerts(df, rx, labs)
        top = alerts[0] if alerts else {}
        rows.append(
            {
                "patient_id": pid,
                "last_checkin": last_dt,
                "risk": (top.get("headline") or "")[:70],
                "active_meds": len((rx.get("active_medications") or []) if isinstance(rx.get("active_medications"), list) else []),
                "abnormal_labs": len((labs.get("abnormal_labs") or []) if isinstance(labs.get("abnormal_labs"), list) else []),
                "rx_risk_flags": ", ".join((rx.get("risk_flags") or []) if isinstance(rx.get("risk_flags"), list) else []),
                "lab_risk_flags": ", ".join((labs.get("risk_flags") or []) if isinstance(labs.get("risk_flags"), list) else []),
            }
        )

    st.subheader("Patients Overview")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pick = st.selectbox("Open patient", patients, index=0 if patients else None)
    if pick:
        st.divider()
        render_doctor_patient_view(pick)


def main() -> None:
    st.set_page_config(page_title="AI Summary Viewer", layout="wide")
    st.title("AI Summary Viewer")

    mode = st.radio("View", ["Patient", "Doctor"], horizontal=True)
    patients = list_patients()
    if not patients:
        st.warning("No patients found in Firestore.")
        return

    if mode == "Doctor":
        render_doctor_view()
        return

    patient_id = st.selectbox("Patient", patients, index=0)
    render_patient_view(patient_id)


if __name__ == "__main__":
    main()
