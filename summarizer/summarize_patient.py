import argparse
import logging
import math
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from google.genai import types

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import load_env_file, require_env


logger = logging.getLogger("summarizer")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_status(v: Any) -> str:
    return str(v or "").strip().lower()


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / norm for x in vec]


def dot(a: List[float], b: List[float]) -> float:
    return float(sum((x * y) for x, y in zip(a, b)))


def parse_bp(bp: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        s = str(bp or "")
        a, b = s.split("/", 1)
        return float(a.strip()), float(b.strip())
    except Exception:
        return None, None


def parse_float(v: Any) -> Optional[float]:
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


def extract_date_from_checkin(data: Dict[str, Any]) -> str:
    v = data.get("date")
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]
    v = data.get("Date")
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]
    v = data.get("createdAt")
    if isinstance(v, str) and len(v) >= 10:
        return v[:10]
    vitals = data.get("vitals")
    if isinstance(vitals, list) and vitals and isinstance(vitals[0], dict):
        measured_at = vitals[0].get("measuredAt")
        if isinstance(measured_at, str) and len(measured_at) >= 10:
            return measured_at[:10]
    return ""


def load_patient_daily_checkins(
    db: firestore.Client,
    patient_id: str,
    backfill_date: bool,
    base_date_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    col = db.collection("Patients").document(patient_id).collection("DailyCheckIns")

    for doc in col.stream():
        data = doc.to_dict() or {}
        dt = extract_date_from_checkin(data)
        out.append({**data, "_doc_id": doc.id, "_date": dt})

    def parse_ymd(s: str) -> Optional[date]:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    out.sort(key=lambda d: (str(d.get("_date") or ""), str(d.get("createdAt") or ""), str(d.get("_doc_id") or "")))
    if not out:
        return []

    updated = 0
    if backfill_date:
        valid_dates = [parse_ymd(str(d.get("_date") or "")) for d in out]
        valid_dates = [d for d in valid_dates if d]
        unique_dates = {str(d.get("_date") or "") for d in out if str(d.get("_date") or "")}

        needs_resequence = (len(unique_dates) != len(out)) or (len(unique_dates) == 0)
        if needs_resequence:
            hinted = parse_ymd(str(base_date_hint or ""))
            base = hinted or (min(valid_dates) if valid_dates else (datetime.now(timezone.utc).date() - timedelta(days=len(out) - 1)))
            batch = db.batch()
            in_batch = 0
            for idx, item in enumerate(out):
                new_dt = (base + timedelta(days=idx)).isoformat()
                if str(item.get("date") or "")[:10] != new_dt:
                    batch.set(
                        col.document(str(item.get("_doc_id") or "")),
                        {"date": new_dt},
                        merge=True,
                    )
                    in_batch += 1
                    updated += 1
                    if in_batch >= 450:
                        batch.commit()
                        batch = db.batch()
                        in_batch = 0
                item["_date"] = new_dt
            if in_batch:
                batch.commit()
        else:
            batch = db.batch()
            in_batch = 0
            for item in out:
                dt = str(item.get("_date") or "")
                if not dt:
                    continue
                if str(item.get("date") or "")[:10] != dt:
                    batch.set(
                        col.document(str(item.get("_doc_id") or "")),
                        {"date": dt},
                        merge=True,
                    )
                    in_batch += 1
                    updated += 1
                    if in_batch >= 450:
                        batch.commit()
                        batch = db.batch()
                        in_batch = 0
            if in_batch:
                batch.commit()

        out.sort(key=lambda d: (str(d.get("_date") or ""), str(d.get("_doc_id") or "")))

    out = [d for d in out if str(d.get("_date") or "")]
    logger.info("daily_checkins_loaded patientId=%s count=%s date_backfilled=%s", patient_id, len(out), updated)
    return out


def load_patient_dated_docs(
    db: firestore.Client,
    patient_id: str,
    collection_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    col = db.collection("Patients").document(patient_id).collection(collection_name)
    for doc in col.stream():
        data = doc.to_dict() or {}
        if normalize_status(data.get("status")) not in ("", "uploaded"):
            continue
        dt = str(data.get("date") or data.get("Date") or "").strip()[:10]
        if not dt:
            continue
        out.append(
            {
                "date": dt,
                "doc_id": doc.id,
                "doc_ref": doc.reference,
                "fileName": str(data.get("fileName") or "").strip(),
                "doctorId": str(data.get("doctorId") or "").strip(),
                "sessionId": str(data.get("sessionId") or "").strip(),
            }
        )
    out.sort(key=lambda d: str(d.get("date") or ""))
    return out


def pick_latest_on_or_before(docs: List[Dict[str, Any]], target_date: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for d in docs:
        dt = str(d.get("date") or "")
        if dt and dt <= target_date:
            best = d
        elif dt and dt > target_date:
            break
    return best


def compute_vitals_summary(checkins: List[Dict[str, Any]], window_days: int) -> Dict[str, Any]:
    if not checkins:
        return {}

    recent = checkins[-window_days:] if window_days > 0 else checkins
    today = recent[-1]
    yesterday = recent[-2] if len(recent) >= 2 else None
    baseline = recent[0]

    def get_vitals(d: Dict[str, Any]) -> Dict[str, Any]:
        vitals = (d.get("vitals") or [])
        if isinstance(vitals, list) and vitals and isinstance(vitals[0], dict):
            return vitals[0]
        return {}

    def avg(values: List[Optional[float]]) -> Optional[float]:
        xs = [x for x in values if isinstance(x, (int, float))]
        if not xs:
            return None
        return float(sum(xs)) / float(len(xs))

    def metric_series(key: str) -> List[Optional[float]]:
        out_vals: List[Optional[float]] = []
        for d in recent:
            v = get_vitals(d).get(key)
            out_vals.append(parse_float(v))
        return out_vals

    def bp_series(part: str) -> List[Optional[float]]:
        out_vals: List[Optional[float]] = []
        for d in recent:
            sbp, dbp = parse_bp(get_vitals(d).get("BP"))
            out_vals.append(sbp if part == "SBP" else dbp)
        return out_vals

    def point_value(d: Dict[str, Any], key: str) -> Optional[float]:
        v = get_vitals(d).get(key)
        return parse_float(v)

    def bp_point(d: Dict[str, Any], part: str) -> Optional[float]:
        sbp, dbp = parse_bp(get_vitals(d).get("BP"))
        return sbp if part == "SBP" else dbp

    last7 = recent[-7:]
    last30 = recent[-30:]

    def avg_for(seq: List[Dict[str, Any]], key: str) -> Optional[float]:
        if key in ("SBP", "DBP"):
            vals = []
            for d in seq:
                vals.append(bp_point(d, key))
            return avg(vals)
        vals = []
        for d in seq:
            vals.append(point_value(d, key))
        return avg(vals)

    def val_for(d: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not d:
            return None
        if key in ("SBP", "DBP"):
            return bp_point(d, key)
        return point_value(d, key)

    keys = ["SBP", "DBP", "HR", "RR", "SpO2", "Temp", "Pulse", "weightKg", "BMI"]
    summary: Dict[str, Any] = {
        "todayDate": str(today.get("_date") or ""),
        "baselineDate": str(baseline.get("_date") or ""),
        "metrics": {},
    }

    for k in keys:
        b = val_for(baseline, k)
        t = val_for(today, k)
        y = val_for(yesterday, k)
        avg7 = avg_for(last7, k)
        avg30 = avg_for(last30, k)
        delta_from_baseline = (t - b) if isinstance(t, (int, float)) and isinstance(b, (int, float)) else None
        summary["metrics"][k] = {
            "today": t,
            "yesterday": y,
            "baseline": b,
            "avg7d": avg7,
            "avg30d": avg30,
            "deltaFromBaseline": delta_from_baseline,
        }

    return summary


def compute_doctor_notes_summary(checkins: List[Dict[str, Any]], window_days: int) -> Dict[str, Any]:
    if not checkins:
        return {}
    recent = checkins[-window_days:] if window_days > 0 else checkins
    today = recent[-1]
    yesterday = recent[-2] if len(recent) >= 2 else None

    def short_note(d: Dict[str, Any]) -> str:
        notes = d.get("Notes") or {}
        if isinstance(notes, dict):
            s = str(notes.get("short") or "").strip()
            if s:
                return s
            f = str(notes.get("full") or "").strip()
            return f
        return ""

    last7 = recent[-7:]
    last30 = recent[-30:]

    def pack(seq: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out = []
        for d in seq:
            s = short_note(d)
            if not s:
                continue
            out.append({"date": str(d.get("_date") or ""), "note": s})
        return out

    return {
        "todayDate": str(today.get("_date") or ""),
        "yesterdayDate": str((yesterday or {}).get("_date") or ""),
        "today": short_note(today),
        "yesterday": short_note(yesterday) if yesterday else "",
        "last7d": pack(last7),
        "last30d": pack(last30),
    }


def load_patient_vectors(
    db: firestore.Client,
    patient_id: str,
    collection_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = db.collection("Patients").document(patient_id).collection(collection_name)
    for parent in base.stream():
        parent_data = parent.to_dict() or {}
        if normalize_status(parent_data.get("status")) not in ("", "uploaded"):
            continue
        for vec in parent.reference.collection("Vectors").select(
            ["chunk_text", "embedding", "embedding_dimension", "page_index", "start_char", "end_char", "chunk_id"]
        ).stream():
            v = vec.to_dict() or {}
            emb = v.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            out.append(
                {
                    "chunk_text": str(v.get("chunk_text") or ""),
                    "embedding": emb,
                    "embedding_dimension": int(v.get("embedding_dimension") or 0),
                    "page_index": v.get("page_index"),
                    "chunk_id": str(v.get("chunk_id") or vec.id),
                    "document_path": parent.reference.path,
                    "date": str(parent_data.get("date") or parent_data.get("Date") or "").strip()[:10],
                    "fileName": str(parent_data.get("fileName") or "").strip(),
                }
            )
    return out


def embed_query(client: genai.Client, text: str, output_dim: int) -> List[float]:
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=output_dim),
    )
    vec = res.embeddings[0].values
    return l2_normalize([float(x) for x in vec])


def top_k_contexts(query_vec: List[float], vectors: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for v in vectors:
        emb = v.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            emb_f = [float(x) for x in emb]
        except Exception:
            continue
        scored.append((dot(query_vec, l2_normalize(emb_f)), v))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for score, v in scored[: max(0, int(top_k))]:
        out.append({**v, "score": float(score)})
    return out


def generate_json_summary(
    client: genai.Client,
    model: str,
    instruction: str,
    contexts: List[Dict[str, Any]],
    schema: Dict[str, Any],
    prior_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx_lines = []
    for c in contexts:
        text = str(c.get("chunk_text") or "").strip()
        if not text:
            continue
        ctx_lines.append(
            {
                "score": float(c.get("score") or 0.0),
                "date": str(c.get("date") or ""),
                "fileName": str(c.get("fileName") or ""),
                "document_path": str(c.get("document_path") or ""),
                "text": text,
            }
        )

    prior = prior_summary or {}
    prompt = (
        "You are a clinical summarizer.\n"
        "Use ONLY the provided context. If information is missing, use null or empty arrays.\n"
        "Return valid JSON matching the provided schema.\n\n"
        f"Task:\n{instruction}\n\n"
        f"Prior summary JSON (may be empty):\n{prior}\n\n"
        f"Context JSON:\n{ctx_lines}\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    if getattr(resp, "parsed", None) is not None:
        return dict(resp.parsed)
    import json

    return json.loads(resp.text or "{}")


def prescription_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "required": ["active_medications", "inactive_medications", "confidence"],
        "properties": {
            "active_medications": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "STRING"},
                        "dose": {"type": "STRING"},
                        "frequency": {"type": "STRING"},
                        "route": {"type": "STRING"},
                        "start_date": {"type": "STRING"},
                        "intended_duration": {"type": "STRING"},
                        "indication": {"type": "STRING"},
                        "source_doc_id": {"type": "STRING"},
                    },
                },
            },
            "inactive_medications": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "STRING"},
                        "dose": {"type": "STRING"},
                        "frequency": {"type": "STRING"},
                        "route": {"type": "STRING"},
                        "start_date": {"type": "STRING"},
                        "end_date": {"type": "STRING"},
                        "reason_stopped": {"type": "STRING"},
                        "source_doc_id": {"type": "STRING"},
                    },
                },
            },
            "confidence": {"type": "NUMBER"},
        },
    }


def lab_report_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "required": ["abnormal_labs", "resolved_labs", "data_quality", "confidence"],
        "properties": {
            "abnormal_labs": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["test_name"],
                    "properties": {
                        "test_name": {"type": "STRING"},
                        "current_value": {"type": "STRING"},
                        "unit": {"type": "STRING"},
                        "baseline_value": {"type": "STRING"},
                        "normal_range": {"type": "STRING"},
                        "trend": {"type": "STRING"},
                        "first_detected_on": {"type": "STRING"},
                        "source_doc_id": {"type": "STRING"},
                    },
                },
            },
            "resolved_labs": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": ["test_name"],
                    "properties": {
                        "test_name": {"type": "STRING"},
                        "previous_high": {"type": "STRING"},
                        "current_value": {"type": "STRING"},
                        "resolution_date": {"type": "STRING"},
                        "source_doc_id": {"type": "STRING"},
                    },
                },
            },
            "data_quality": {
                "type": "OBJECT",
                "required": ["fasting_status_known", "missing_comparative_tests"],
                "properties": {
                    "fasting_status_known": {"type": "BOOLEAN"},
                    "missing_comparative_tests": {"type": "BOOLEAN"},
                },
            },
            "confidence": {"type": "NUMBER"},
        },
    }


def slug_doc_id(prefix: str, date_str: str, fallback: str) -> str:
    dt = str(date_str or "").strip()[:10]
    if len(dt) == 10 and dt[4] == "-" and dt[7] == "-":
        return f"{prefix}_{dt.replace('-', '_')}"
    fb = str(fallback or "").strip()
    return fb if fb else prefix


def _safe_list(v: Any) -> List[Dict[str, Any]]:
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return []


def _norm_name(v: Any) -> str:
    return " ".join(str(v or "").strip().lower().split())


def compute_med_delta(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    prev_active = _safe_list((prev or {}).get("active_medications"))
    curr_active = _safe_list((curr or {}).get("active_medications"))
    curr_inactive = _safe_list((curr or {}).get("inactive_medications"))

    prev_map = {_norm_name(m.get("name")): m for m in prev_active if _norm_name(m.get("name"))}
    curr_map = {_norm_name(m.get("name")): m for m in curr_active if _norm_name(m.get("name"))}
    inactive_names = {_norm_name(m.get("name")) for m in curr_inactive if _norm_name(m.get("name"))}

    types: List[str] = []
    for n in curr_map:
        if n not in prev_map:
            types.append("new_medication")
        else:
            a = prev_map.get(n) or {}
            b = curr_map.get(n) or {}
            if _norm_name(a.get("dose")) != _norm_name(b.get("dose")) or _norm_name(a.get("frequency")) != _norm_name(b.get("frequency")):
                types.append("dose_change")
    for n in prev_map:
        if (n not in curr_map) and (n in inactive_names or n not in curr_map):
            types.append("stopped_medication")

    uniq_types = []
    seen = set()
    for t in types:
        if t in seen:
            continue
        seen.add(t)
        uniq_types.append(t)

    since = str((prev or {}).get("as_of_date") or "")
    return {"has_changes": bool(uniq_types), "types": uniq_types, "since": since}


def risk_flags_for_meds(summary: Dict[str, Any]) -> List[str]:
    active = _safe_list((summary or {}).get("active_medications"))
    names = [_norm_name(m.get("name")) for m in active]
    flags: List[str] = []
    if any("metformin" in n for n in names):
        flags.append("long_term_metformin")
    n = len(active)
    if n >= 5:
        flags.append("polypharmacy_risk_high")
    elif n >= 3:
        flags.append("polypharmacy_risk_medium")
    else:
        flags.append("polypharmacy_risk_low")
    return flags


def finalize_prescription_summary(
    extracted: Dict[str, Any],
    patient_id: str,
    as_of_date: str,
    generated_at: str,
    source_doc_id: str,
    prev_summary: Dict[str, Any],
) -> Dict[str, Any]:
    active = _safe_list((extracted or {}).get("active_medications"))
    inactive = _safe_list((extracted or {}).get("inactive_medications"))
    for m in active:
        if not str(m.get("source_doc_id") or "").strip():
            m["source_doc_id"] = source_doc_id
    for m in inactive:
        if not str(m.get("source_doc_id") or "").strip():
            m["source_doc_id"] = source_doc_id

    curr = {"active_medications": active, "inactive_medications": inactive, "as_of_date": as_of_date}
    delta = compute_med_delta(prev_summary or {}, curr)
    out = {
        "patient_id": patient_id,
        "as_of_date": as_of_date,
        "active_medications": active,
        "inactive_medications": inactive,
        "delta": delta,
        "risk_flags": risk_flags_for_meds(curr),
        "confidence": float((extracted or {}).get("confidence") or 0.0),
        "generated_at": generated_at,
    }
    return out


def _to_float_or_none(v: Any) -> Optional[float]:
    x = parse_float(v)
    return float(x) if isinstance(x, (int, float)) else None


def compute_lab_delta(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    prev_abn = _safe_list((prev or {}).get("abnormal_labs"))
    curr_abn = _safe_list((curr or {}).get("abnormal_labs"))
    prev_names = {_norm_name(x.get("test_name")) for x in prev_abn if _norm_name(x.get("test_name"))}
    curr_names = {_norm_name(x.get("test_name")) for x in curr_abn if _norm_name(x.get("test_name"))}
    types: List[str] = []
    if any(n and n not in prev_names for n in curr_names):
        types.append("new_abnormality")
    if any(n and n not in curr_names for n in prev_names):
        types.append("resolved_abnormality")
    since = str((prev or {}).get("as_of_date") or "")
    return {"has_changes": bool(types), "types": types, "since": since}


def risk_flags_for_labs(summary: Dict[str, Any]) -> List[str]:
    abn = _safe_list((summary or {}).get("abnormal_labs"))
    flags: List[str] = []
    for x in abn:
        name = _norm_name(x.get("test_name"))
        days_abn = parse_float(x.get("days_abnormal"))
        trend = _norm_name(x.get("trend"))
        if "lactic" in name and isinstance(days_abn, (int, float)) and days_abn >= 3:
            flags.append("persistent_lactic_acidosis")
        if trend == "worsening" and isinstance(days_abn, (int, float)) and days_abn >= 3:
            break
    return flags


def finalize_lab_report_summary(
    extracted: Dict[str, Any],
    patient_id: str,
    as_of_date: str,
    generated_at: str,
    source_doc_id: str,
    prev_summary: Dict[str, Any],
) -> Dict[str, Any]:
    prev_abn = _safe_list((prev_summary or {}).get("abnormal_labs"))
    prev_map = {_norm_name(x.get("test_name")): x for x in prev_abn if _norm_name(x.get("test_name"))}

    abnormal = _safe_list((extracted or {}).get("abnormal_labs"))
    for x in abnormal:
        test = _norm_name(x.get("test_name"))
        if not str(x.get("source_doc_id") or "").strip():
            x["source_doc_id"] = source_doc_id
        prev_item = prev_map.get(test) or {}
        if not str(x.get("first_detected_on") or "").strip():
            x["first_detected_on"] = str(prev_item.get("first_detected_on") or as_of_date)
        prev_days = parse_float(prev_item.get("days_abnormal"))
        if isinstance(prev_days, (int, float)) and prev_item:
            x["days_abnormal"] = int(prev_days) + 1
        else:
            x["days_abnormal"] = 1
        cur_v = _to_float_or_none(x.get("current_value"))
        prev_cur = _to_float_or_none(prev_item.get("current_value"))
        if cur_v is not None and prev_cur is not None:
            if cur_v > prev_cur:
                x["trend"] = "worsening"
            elif cur_v < prev_cur:
                x["trend"] = "improving"
            else:
                x["trend"] = "stable"
        x["current_value"] = cur_v
        x["baseline_value"] = _to_float_or_none(x.get("baseline_value") or prev_item.get("baseline_value"))

    resolved = _safe_list((extracted or {}).get("resolved_labs"))
    for x in resolved:
        if not str(x.get("source_doc_id") or "").strip():
            x["source_doc_id"] = source_doc_id
        x["previous_high"] = _to_float_or_none(x.get("previous_high"))
        x["current_value"] = _to_float_or_none(x.get("current_value"))

    dq = (extracted or {}).get("data_quality") or {}
    if not isinstance(dq, dict):
        dq = {}
    missing_comp = any(x.get("baseline_value") is None for x in abnormal) if abnormal else False
    dq_out = {
        "fasting_status_known": bool(dq.get("fasting_status_known") or False),
        "missing_comparative_tests": bool(dq.get("missing_comparative_tests") or missing_comp),
    }

    curr = {"abnormal_labs": abnormal, "as_of_date": as_of_date}
    delta = compute_lab_delta(prev_summary or {}, curr)
    out = {
        "patient_id": patient_id,
        "as_of_date": as_of_date,
        "abnormal_labs": abnormal,
        "resolved_labs": resolved,
        "delta": delta,
        "risk_flags": risk_flags_for_labs(curr),
        "data_quality": dq_out,
        "confidence": float((extracted or {}).get("confidence") or 0.0),
        "generated_at": generated_at,
    }
    return out


def carry_forward_summary(
    prev: Dict[str, Any],
    patient_id: str,
    as_of_date: str,
    generated_at: str,
    kind: str,
) -> Dict[str, Any]:
    prev_date = str((prev or {}).get("as_of_date") or "")
    if kind == "prescriptions":
        base = {
            "patient_id": patient_id,
            "as_of_date": as_of_date,
            "active_medications": _safe_list((prev or {}).get("active_medications")),
            "inactive_medications": _safe_list((prev or {}).get("inactive_medications")),
            "confidence": float((prev or {}).get("confidence") or 0.0),
            "generated_at": generated_at,
            "delta": {"has_changes": False, "types": [], "since": prev_date},
        }
        base["risk_flags"] = risk_flags_for_meds(base)
        return base
    if kind == "lab_reports":
        dq = (prev or {}).get("data_quality") or {}
        if not isinstance(dq, dict):
            dq = {}
        base = {
            "patient_id": patient_id,
            "as_of_date": as_of_date,
            "abnormal_labs": _safe_list((prev or {}).get("abnormal_labs")),
            "resolved_labs": _safe_list((prev or {}).get("resolved_labs")),
            "data_quality": {
                "fasting_status_known": bool(dq.get("fasting_status_known") or False),
                "missing_comparative_tests": bool(dq.get("missing_comparative_tests") or False),
            },
            "confidence": float((prev or {}).get("confidence") or 0.0),
            "generated_at": generated_at,
            "delta": {"has_changes": False, "types": [], "since": prev_date},
        }
        base["risk_flags"] = risk_flags_for_labs(base)
        return base
    return {}


def summarize_prescriptions(
    db: firestore.Client,
    embed_client: genai.Client,
    llm_client: genai.Client,
    patient_id: str,
    top_k: int,
    llm_model: str,
    output_dim: int,
) -> Dict[str, Any]:
    vectors = load_patient_vectors(db=db, patient_id=patient_id, collection_name="Prescriptions")
    return summarize_prescriptions_from_vectors(
        embed_client=embed_client,
        llm_client=llm_client,
        vectors=vectors,
        top_k=top_k,
        llm_model=llm_model,
        output_dim=output_dim,
        prior_summary=None,
        patient_id=patient_id,
        as_of_date=iso_utc_now()[:10],
        source_doc_id="rx",
    )


def summarize_lab_reports(
    db: firestore.Client,
    embed_client: genai.Client,
    llm_client: genai.Client,
    patient_id: str,
    top_k: int,
    llm_model: str,
    output_dim: int,
) -> Dict[str, Any]:
    vectors = load_patient_vectors(db=db, patient_id=patient_id, collection_name="LabReports")
    return summarize_lab_reports_from_vectors(
        embed_client=embed_client,
        llm_client=llm_client,
        vectors=vectors,
        top_k=top_k,
        llm_model=llm_model,
        output_dim=output_dim,
        prior_summary=None,
        patient_id=patient_id,
        as_of_date=iso_utc_now()[:10],
        source_doc_id="lab",
    )


def load_vectors_for_patient_doc(
    doc_ref: firestore.DocumentReference,
    parent_date: str,
    parent_file_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for vec in doc_ref.collection("Vectors").select(
        ["chunk_text", "embedding", "embedding_dimension", "page_index", "start_char", "end_char", "chunk_id"]
    ).stream():
        v = vec.to_dict() or {}
        emb = v.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        out.append(
            {
                "chunk_text": str(v.get("chunk_text") or ""),
                "embedding": emb,
                "embedding_dimension": int(v.get("embedding_dimension") or 0),
                "page_index": v.get("page_index"),
                "chunk_id": str(v.get("chunk_id") or vec.id),
                "document_path": doc_ref.path,
                "date": str(parent_date or ""),
                "fileName": str(parent_file_name or ""),
            }
        )
    return out


def summarize_prescriptions_from_vectors(
    embed_client: genai.Client,
    llm_client: genai.Client,
    vectors: List[Dict[str, Any]],
    top_k: int,
    llm_model: str,
    output_dim: int,
    prior_summary: Optional[Dict[str, Any]],
    patient_id: str,
    as_of_date: str,
    source_doc_id: str,
) -> Dict[str, Any]:
    if not vectors:
        return {}

    queries = [
        "Extract active medications as structured objects.",
        "Extract inactive/stopped medications as structured objects.",
    ]

    contexts: List[Dict[str, Any]] = []
    for q in queries:
        qv = embed_query(embed_client, q, output_dim=output_dim)
        contexts.extend(top_k_contexts(qv, vectors, top_k=top_k))

    contexts.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for c in contexts:
        key = (c.get("document_path"), c.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
        if len(dedup) >= max(10, top_k * 4):
            break

    extracted = generate_json_summary(
        client=llm_client,
        model=llm_model,
        instruction=(
            "Build an as-of medication state for the given patient/date.\n"
            "Carry forward medications from prior summary unless changed/stopped by the current context.\n"
            f"Set source_doc_id to '{source_doc_id}' when missing.\n"
            "Return only: active_medications, inactive_medications, confidence."
        ),
        contexts=dedup,
        schema=prescription_schema(),
        prior_summary=prior_summary,
    )
    return finalize_prescription_summary(
        extracted=extracted,
        patient_id=patient_id,
        as_of_date=as_of_date,
        generated_at=iso_utc_now(),
        source_doc_id=source_doc_id,
        prev_summary=dict(prior_summary or {}),
    )


def summarize_lab_reports_from_vectors(
    embed_client: genai.Client,
    llm_client: genai.Client,
    vectors: List[Dict[str, Any]],
    top_k: int,
    llm_model: str,
    output_dim: int,
    prior_summary: Optional[Dict[str, Any]],
    patient_id: str,
    as_of_date: str,
    source_doc_id: str,
) -> Dict[str, Any]:
    if not vectors:
        return {}

    queries = [
        "Extract abnormal lab tests with values, units, and ranges if present.",
        "Extract labs that resolved with prior high and current value if present.",
    ]

    contexts: List[Dict[str, Any]] = []
    for q in queries:
        qv = embed_query(embed_client, q, output_dim=output_dim)
        contexts.extend(top_k_contexts(qv, vectors, top_k=top_k))

    contexts.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for c in contexts:
        key = (c.get("document_path"), c.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
        if len(dedup) >= max(10, top_k * 4):
            break

    extracted = generate_json_summary(
        client=llm_client,
        model=llm_model,
        instruction=(
            "Build an as-of lab state for the given patient/date.\n"
            "Carry forward abnormalities from prior summary unless resolved by the current context.\n"
            f"Set source_doc_id to '{source_doc_id}' when missing.\n"
            "Return only: abnormal_labs, resolved_labs, data_quality, confidence."
        ),
        contexts=dedup,
        schema=lab_report_schema(),
        prior_summary=prior_summary,
    )
    return finalize_lab_report_summary(
        extracted=extracted,
        patient_id=patient_id,
        as_of_date=as_of_date,
        generated_at=iso_utc_now(),
        source_doc_id=source_doc_id,
        prev_summary=dict(prior_summary or {}),
    )


def run_incremental_summaries(
    db: firestore.Client,
    embed_client: genai.Client,
    llm_client: genai.Client,
    patient_id: str,
    pipeline: str,
    days: int,
    top_k: int,
    llm_model: str,
    output_dim: int,
    out_collection: str,
) -> None:
    prescriptions = load_patient_dated_docs(db=db, patient_id=patient_id, collection_name="Prescriptions")
    lab_reports = load_patient_dated_docs(db=db, patient_id=patient_id, collection_name="LabReports")

    base_hint = ""
    all_dates = [str(d.get("date") or "") for d in (prescriptions + lab_reports)]
    all_dates = [d for d in all_dates if len(d) >= 10]
    if all_dates:
        base_hint = min(all_dates)

    checkins = load_patient_daily_checkins(db=db, patient_id=patient_id, backfill_date=True, base_date_hint=base_hint)
    if not checkins:
        logger.info("no_checkins patientId=%s", patient_id)
        return

    days = max(1, int(days))
    days = min(days, len(checkins))
    checkins = checkins[:days]

    prev_doc_id = ""
    prev_prescription_doc_path = ""
    prev_lab_doc_path = ""
    prev_prescription_summary: Dict[str, Any] = {}
    prev_lab_report_summary: Dict[str, Any] = {}

    for idx in range(days):
        window = checkins[: idx + 1]
        latest = window[-1] if window else {}
        day = str(latest.get("_date") or "")
        doctor_id = str(latest.get("doctorId") or "")
        generated_at = iso_utc_now()

        daily_summary = compute_vitals_summary(window, window_days=len(window))
        notes_summary = compute_doctor_notes_summary(window, window_days=len(window))

        prescription_summary: Dict[str, Any] = {}
        lab_report_summary: Dict[str, Any] = {}

        if pipeline in ("12", "123"):
            picked = pick_latest_on_or_before(prescriptions, day)
            if not picked:
                prescription_summary = carry_forward_summary(
                    prev=prev_prescription_summary,
                    patient_id=patient_id,
                    as_of_date=day,
                    generated_at=generated_at,
                    kind="prescriptions",
                )
            else:
                doc_ref = picked.get("doc_ref")
                doc_path = str(getattr(doc_ref, "path", "") or "")
                if doc_path and doc_path == prev_prescription_doc_path:
                    prescription_summary = carry_forward_summary(
                        prev=prev_prescription_summary,
                        patient_id=patient_id,
                        as_of_date=day,
                        generated_at=generated_at,
                        kind="prescriptions",
                    )
                else:
                    vectors = load_vectors_for_patient_doc(
                        doc_ref=doc_ref,
                        parent_date=str(picked.get("date") or ""),
                        parent_file_name=str(picked.get("fileName") or ""),
                    )
                    if vectors:
                        source_doc_id = slug_doc_id("rx", str(picked.get("date") or ""), str(picked.get("doc_id") or ""))
                        prescription_summary = summarize_prescriptions_from_vectors(
                            embed_client=embed_client,
                            llm_client=llm_client,
                            vectors=vectors,
                            top_k=top_k,
                            llm_model=llm_model,
                            output_dim=output_dim,
                            prior_summary=prev_prescription_summary,
                            patient_id=patient_id,
                            as_of_date=day,
                            source_doc_id=source_doc_id,
                        )
                        prev_prescription_doc_path = doc_path
                    else:
                        prescription_summary = carry_forward_summary(
                            prev=prev_prescription_summary,
                            patient_id=patient_id,
                            as_of_date=day,
                            generated_at=generated_at,
                            kind="prescriptions",
                        )

        if pipeline == "123":
            picked = pick_latest_on_or_before(lab_reports, day)
            if not picked:
                lab_report_summary = carry_forward_summary(
                    prev=prev_lab_report_summary,
                    patient_id=patient_id,
                    as_of_date=day,
                    generated_at=generated_at,
                    kind="lab_reports",
                )
            else:
                doc_ref = picked.get("doc_ref")
                doc_path = str(getattr(doc_ref, "path", "") or "")
                if doc_path and doc_path == prev_lab_doc_path:
                    lab_report_summary = carry_forward_summary(
                        prev=prev_lab_report_summary,
                        patient_id=patient_id,
                        as_of_date=day,
                        generated_at=generated_at,
                        kind="lab_reports",
                    )
                else:
                    vectors = load_vectors_for_patient_doc(
                        doc_ref=doc_ref,
                        parent_date=str(picked.get("date") or ""),
                        parent_file_name=str(picked.get("fileName") or ""),
                    )
                    if vectors:
                        source_doc_id = slug_doc_id("lab", str(picked.get("date") or ""), str(picked.get("doc_id") or ""))
                        lab_report_summary = summarize_lab_reports_from_vectors(
                            embed_client=embed_client,
                            llm_client=llm_client,
                            vectors=vectors,
                            top_k=top_k,
                            llm_model=llm_model,
                            output_dim=output_dim,
                            prior_summary=prev_lab_report_summary,
                            patient_id=patient_id,
                            as_of_date=day,
                            source_doc_id=source_doc_id,
                        )
                        prev_lab_doc_path = doc_path
                    else:
                        lab_report_summary = carry_forward_summary(
                            prev=prev_lab_report_summary,
                            patient_id=patient_id,
                            as_of_date=day,
                            generated_at=generated_at,
                            kind="lab_reports",
                        )

        doc_id = f"{patient_id}_{day}_p{pipeline}"
        payload = {
            "patientId": patient_id,
            "doctorId": doctor_id,
            "date": day,
            "dayIndex": idx + 1,
            "pipeline": pipeline,
            "previousDocId": prev_doc_id,
            "dailyCheckinSummary": daily_summary,
            "doctorNotesSummary": notes_summary,
            "prescriptionSummary": prescription_summary,
            "labReportSummary": lab_report_summary,
            "updatedAt": iso_utc_now(),
        }
        write_summary_doc(db=db, collection=out_collection, doc_id=doc_id, payload=payload)
        logger.info("summary_written patientId=%s date=%s doc_id=%s", patient_id, day, doc_id)

        prev_doc_id = doc_id
        prev_prescription_summary = dict(prescription_summary or {})
        prev_lab_report_summary = dict(lab_report_summary or {})


def write_summary_doc(
    db: firestore.Client,
    collection: str,
    doc_id: str,
    payload: Dict[str, Any],
) -> None:
    db.collection(collection).document(doc_id).set(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--pipeline", default="123", choices=["1", "12", "123"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--summary-model", default="gemini-2.0-flash")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--out-collection", default="Summaries")
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    load_env_file(project_root / ".env")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str((project_root / args.service_account).resolve())))
    db = firestore.client()

    google_key = require_env("GOOGLE_API_KEY")
    embed_client = genai.Client(api_key=google_key)
    llm_client = embed_client

    patient_id = str(args.patient_id).strip()
    window_days = max(1, int(args.days))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{patient_id}_{run_id}_{uuid.uuid4().hex[:8]}"
    out_collection = str(args.out_collection).strip() or "Summaries"

    start = time.perf_counter()
    logger.info("run_start patientId=%s pipeline=%s sessionId=%s", patient_id, args.pipeline, session_id)

    run_incremental_summaries(
        db=db,
        embed_client=embed_client,
        llm_client=llm_client,
        patient_id=patient_id,
        pipeline=str(args.pipeline),
        days=window_days,
        top_k=max(1, int(args.top_k)),
        llm_model=str(args.summary_model),
        output_dim=int(args.embedding_dim),
        out_collection=out_collection,
    )

    logger.info("run_done duration_ms=%s out_collection=%s", int((time.perf_counter() - start) * 1000), out_collection)


if __name__ == "__main__":
    main()
