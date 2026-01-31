import argparse
import json
import math
import random
import re
import uuid
from urllib.parse import quote
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import firebase_admin
from firebase_admin import credentials, firestore, storage
from google import genai
from google.genai import types

from main import load_env_file, require_env


def init_default_vitals(db: firestore.Client, collection: str, doc_id: str) -> None:
    payload = {
        "user_id": "user_123",
        "date": "2023-01-01",
        "BP": "120/80",
        "HR": 72,
        "RR": 16,
        "SpO2": 98,
        "Temp": 36.8,
        "Pulse": 72,
        "heightCm": 170,
        "weightKg": 70,
        "BMI": 24.2,
        "waistHip": "0.84",
    }
    db.collection(collection).document(doc_id).set(payload, merge=True)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def bmi(height_cm: float, weight_kg: float) -> float:
    h_m = max(0.5, float(height_cm) / 100.0)
    return float(weight_kg) / (h_m * h_m)


def bp_string(sbp: float, dbp: float) -> str:
    return f"{int(round(sbp))}/{int(round(dbp))}"


def _flare(day_index: int, center: int, width: float) -> float:
    z = (day_index - center) / max(1.0, width)
    return math.exp(-(z * z))


def generate_patient_vitals(patient_index: int, day_index: int, total_days: int, rng: random.Random) -> Dict[str, Any]:
    t = 0.0 if total_days <= 1 else day_index / float(total_days - 1)

    if patient_index == 0:
        height_cm = 171.0
        weight_kg = 71.0 + rng.normalvariate(0.0, 0.15)

        temp = 36.8 + 1.8 * math.exp(-day_index / 10.0) + rng.normalvariate(0.0, 0.12)
        hr = 72.0 + 35.0 * math.exp(-day_index / 14.0) + rng.normalvariate(0.0, 2.5)
        rr = 16.0 + 6.0 * math.exp(-day_index / 12.0) + rng.normalvariate(0.0, 1.0)
        spo2 = 98.0 - 2.0 * math.exp(-day_index / 12.0) + rng.normalvariate(0.0, 0.6)

        sbp = 118.0 + 10.0 * math.exp(-day_index / 14.0) + rng.normalvariate(0.0, 2.0)
        dbp = 78.0 + 5.0 * math.exp(-day_index / 16.0) + rng.normalvariate(0.0, 1.5)
        waist_hip = 0.84 + rng.normalvariate(0.0, 0.01)

    elif patient_index == 1:
        height_cm = 175.0
        weight_kg = 98.0 - (10.0 * t) + rng.normalvariate(0.0, 0.25)

        sbp = 150.0 - (20.0 * t) + 2.0 * math.sin(day_index / 7.0) + rng.normalvariate(0.0, 2.5)
        dbp = 95.0 - (10.0 * t) + 1.5 * math.sin(day_index / 9.0) + rng.normalvariate(0.0, 1.8)

        hr = 88.0 - (14.0 * t) + rng.normalvariate(0.0, 2.0)
        rr = 16.0 + rng.normalvariate(0.0, 0.8)
        spo2 = 98.0 + rng.normalvariate(0.0, 0.4)
        temp = 36.7 + rng.normalvariate(0.0, 0.1)
        waist_hip = 0.98 - (0.08 * t) + rng.normalvariate(0.0, 0.01)

    else:
        height_cm = 168.0
        weight_kg = 76.0 - (1.5 * t) + rng.normalvariate(0.0, 0.2)

        flare_1 = _flare(day_index, 50, 5.5)
        flare_2 = _flare(day_index, 130, 6.5)
        flare = (1.0 * flare_1) + (0.8 * flare_2)

        spo2 = 93.0 + (2.0 * t) - (5.0 * flare) + rng.normalvariate(0.0, 0.7)
        rr = 20.0 + (6.0 * flare) + rng.normalvariate(0.0, 1.1)
        hr = 82.0 + (18.0 * flare) + rng.normalvariate(0.0, 2.5)
        temp = 36.8 + (0.5 * flare) + rng.normalvariate(0.0, 0.12)

        sbp = 126.0 + (3.0 * flare) + rng.normalvariate(0.0, 2.0)
        dbp = 82.0 + (2.0 * flare) + rng.normalvariate(0.0, 1.5)
        waist_hip = 0.88 + rng.normalvariate(0.0, 0.01)

    weight_kg = clamp(weight_kg, 40.0, 180.0)
    temp = clamp(temp, 35.6, 40.2)
    hr = clamp(hr, 45.0, 140.0)
    rr = clamp(rr, 10.0, 40.0)
    spo2 = clamp(spo2, 85.0, 100.0)
    sbp = clamp(sbp, 85.0, 210.0)
    dbp = clamp(dbp, 45.0, 140.0)
    waist_hip = clamp(waist_hip, 0.65, 1.20)

    b = bmi(height_cm, weight_kg)
    pulse = hr + rng.normalvariate(0.0, 1.0)

    return {
        "BP": bp_string(sbp, dbp),
        "HR": int(round(hr)),
        "RR": int(round(rr)),
        "SpO2": int(round(spo2)),
        "Temp": round(float(temp), 1),
        "Pulse": int(round(pulse)),
        "heightCm": int(round(height_cm)),
        "weightKg": round(float(weight_kg), 1),
        "BMI": round(float(b), 1),
        "waistHip": f"{waist_hip:.2f}",
    }


def build_seed_ids() -> Tuple[List[str], Dict[str, List[str]]]:
    patient_ids = ["patient_001", "patient_002", "patient_003"]
    doctor_ids_by_patient: Dict[str, List[str]] = {}
    for i, pid in enumerate(patient_ids, start=1):
        doctor_ids_by_patient[pid] = [f"doctor_{i:03d}_{suffix}" for suffix in ("A", "B", "C", "D")]
    return patient_ids, doctor_ids_by_patient


def seed_daily_checkins(
    db: firestore.Client,
    collection: str,
    days: int,
    start_day: date,
) -> int:
    patient_ids, doctors_by_patient = build_seed_ids()
    docs: List[Tuple[str, Dict[str, Any]]] = []

    now = datetime.now(timezone.utc)
    for day_index in range(days):
        d = start_day + timedelta(days=day_index)
        date_str = d.isoformat()
        measured_at = datetime(d.year, d.month, d.day, 9, 0, tzinfo=timezone.utc)

        for patient_index, patient_id in enumerate(patient_ids):
            doctor_id = doctors_by_patient[patient_id][day_index % 4]
            doc_id = f"{patient_id}_{d.strftime('%Y%m%d')}"
            session_id = f"session_{doc_id}"

            rng = random.Random(f"{patient_id}:{doc_id}")
            vitals_item = {
                **generate_patient_vitals(patient_index, day_index, days, rng),
                "measuredAt": iso_utc(measured_at),
            }

            payload: Dict[str, Any] = {
                "Date": date_str,
                "createdAt": iso_utc(now),
                "doctorId": doctor_id,
                "patientId": patient_id,
                "sessionId": session_id,
                "vitals": [vitals_item],
            }
            docs.append((doc_id, payload))

    written = 0
    batch = db.batch()
    in_batch = 0
    for doc_id, payload in docs:
        ref = db.collection(collection).document(doc_id)
        batch.set(ref, payload, merge=True)
        in_batch += 1
        if in_batch >= 450:
            batch.commit()
            written += in_batch
            batch = db.batch()
            in_batch = 0
    if in_batch:
        batch.commit()
        written += in_batch

    return written


def parse_start_day(start_date: str, days: int) -> date:
    if start_date.strip():
        y, m, d = (int(x) for x in start_date.strip().split("-", 2))
        return date(y, m, d)
    return datetime.now(timezone.utc).date() - timedelta(days=days - 1)


def parse_bp(bp: str) -> Tuple[int, int]:
    try:
        s, d = bp.split("/", 1)
        return int(round(float(s.strip()))), int(round(float(d.strip())))
    except Exception:
        return 0, 0


def patient_story(patient_id: str) -> str:
    if patient_id == "patient_001":
        return "Febrile illness early; symptoms and vitals trend back to baseline over ~2 weeks."
    if patient_id == "patient_002":
        return "Lifestyle + meds: weight loss with improving BP/HR over months; needs steady monitoring."
    return "Chronic lung disease with lower baseline SpO2; 2 exacerbations with recovery."


def notes_response_schema() -> Dict[str, Any]:
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "required": ["doc_id", "short", "full", "patient"],
            "properties": {
                "doc_id": {"type": "STRING"},
                "short": {"type": "STRING"},
                "full": {"type": "STRING"},
                "patient": {"type": "STRING"},
            },
        },
    }


def generate_notes_batch(
    client: genai.Client,
    model: str,
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prompt = (
        "You are a doctor writing daily check-in notes based ONLY on provided vitals and trend.\n"
        "Output MUST be valid JSON matching the provided schema. No markdown.\n\n"
        "For each entry return:\n"
        "- doc_id: same as input doc_id\n"
        "- short: clinician shorthand note (<=220 chars)\n"
        "- full: standard doctor note with brief assessment + plan + red flags (<=650 chars)\n"
        "- patient: patient-friendly explanation + advice (<=500 chars)\n\n"
        "Use realistic medical language. Don't invent labs, meds, diagnoses, or symptoms.\n"
        "Do mention trends if delta_yday shows change.\n"
        "If SpO2 <= 92, RR >= 24, HR >= 120, Temp >= 39.0, or BP >= 180/110: add clear escalation advice.\n\n"
        f"Entries JSON:\n{json.dumps(entries, ensure_ascii=False)}\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=notes_response_schema(),
        ),
    )
    if getattr(resp, "parsed", None) is not None:
        return list(resp.parsed)
    return json.loads(resp.text or "[]")


def choose_first_working_model(client: genai.Client, model_candidates: List[str]) -> str:
    for m in model_candidates:
        try:
            _ = client.models.generate_content(
                model=m,
                contents="Return JSON: {\"ok\": true}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "BOOLEAN"}},
                    },
                ),
            )
            return m
        except Exception:
            continue
    raise RuntimeError(f"No usable model found from candidates: {model_candidates}")


def generate_doctor_notes_for_daily_checkins(
    db: firestore.Client,
    client: genai.Client,
    collection: str,
    model: str,
    days: int,
    start_day: date,
    batch_size: int,
    limit: int,
    overwrite: bool,
) -> int:
    patient_ids, _ = build_seed_ids()
    now = datetime.now(timezone.utc)

    pending: List[Dict[str, Any]] = []
    pending_refs: List[firestore.DocumentReference] = []
    total_written = 0

    prev_by_patient: Dict[str, Dict[str, Any]] = {}

    def flush() -> None:
        nonlocal pending, pending_refs, total_written
        if not pending:
            return
        items = generate_notes_batch(client=client, model=model, entries=pending)
        notes_by_doc_id = {str(x.get("doc_id")): x for x in items if isinstance(x, dict)}

        batch = db.batch()
        updated = 0
        for ref in pending_refs:
            doc_id = ref.id
            n = notes_by_doc_id.get(doc_id)
            if not n:
                continue
            payload: Dict[str, Any] = {
                "Notes": {
                    "short": str(n.get("short", "")).strip(),
                    "full": str(n.get("full", "")).strip(),
                    "patient": str(n.get("patient", "")).strip(),
                    "model": model,
                    "generatedAt": iso_utc(now),
                },
                "doctorNotes": firestore.DELETE_FIELD,
            }
            batch.set(ref, payload, merge=True)
            updated += 1
        if updated:
            batch.commit()
            total_written += updated
            print("doctor_notes_progress", total_written)

        pending = []
        pending_refs = []

    for day_index in range(days):
        d = start_day + timedelta(days=day_index)
        for patient_id in patient_ids:
            if limit and total_written >= limit:
                flush()
                return total_written

            doc_id = f"{patient_id}_{d.strftime('%Y%m%d')}"
            ref = db.collection(collection).document(doc_id)
            snap = ref.get()
            if not snap.exists:
                continue

            data = snap.to_dict() or {}
            if (not overwrite) and data.get("Notes"):
                continue

            vitals_arr = data.get("vitals") or []
            vitals = vitals_arr[0] if isinstance(vitals_arr, list) and vitals_arr else {}
            if not isinstance(vitals, dict):
                continue

            bp = str(vitals.get("BP", ""))
            sbp, dbp = parse_bp(bp)
            prev = prev_by_patient.get(patient_id) or {}

            delta_yday: Dict[str, Any] = {}
            if prev:
                prev_sbp, prev_dbp = parse_bp(str(prev.get("BP", "")))
                delta_yday = {
                    "SBP": sbp - prev_sbp,
                    "DBP": dbp - prev_dbp,
                    "HR": int(vitals.get("HR", 0)) - int(prev.get("HR", 0)),
                    "RR": int(vitals.get("RR", 0)) - int(prev.get("RR", 0)),
                    "SpO2": int(vitals.get("SpO2", 0)) - int(prev.get("SpO2", 0)),
                    "Temp": float(vitals.get("Temp", 0.0)) - float(prev.get("Temp", 0.0)),
                    "weightKg": float(vitals.get("weightKg", 0.0)) - float(prev.get("weightKg", 0.0)),
                }

            entry = {
                "doc_id": doc_id,
                "date": str(data.get("Date") or d.isoformat()),
                "day_index": day_index,
                "patient_id": patient_id,
                "doctor_id": str(data.get("doctorId") or ""),
                "story": patient_story(patient_id),
                "vitals": {
                    "BP": bp,
                    "HR": vitals.get("HR"),
                    "RR": vitals.get("RR"),
                    "SpO2": vitals.get("SpO2"),
                    "Temp": vitals.get("Temp"),
                    "Pulse": vitals.get("Pulse"),
                    "heightCm": vitals.get("heightCm"),
                    "weightKg": vitals.get("weightKg"),
                    "BMI": vitals.get("BMI"),
                    "waistHip": vitals.get("waistHip"),
                },
                "delta_yday": delta_yday,
            }

            pending.append(entry)
            pending_refs.append(ref)

            prev_by_patient[patient_id] = dict(entry["vitals"])

            if len(pending) >= batch_size:
                flush()

    flush()
    return total_written


def pdf_escape_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_text_pdf(pdf_path: Path, title: str, content: str) -> None:
    lines = [title.strip(), ""] + [ln.rstrip() for ln in (content or "").splitlines()]
    lines = [ln if ln else "" for ln in lines]

    page_width = 612
    page_height = 792
    margin_left = 48
    margin_top = 64
    font_size = 11
    leading = 14
    max_lines_per_page = max(1, int((page_height - (margin_top + 72)) / leading))

    pages: List[List[str]] = []
    for i in range(0, len(lines), max_lines_per_page):
        pages.append(lines[i : i + max_lines_per_page])
    if not pages:
        pages = [[""]]

    objects: List[bytes] = []

    def add_obj(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    page_obj_ids: List[int] = []
    content_obj_ids: List[int] = []

    font_obj_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_obj_id = add_obj(b"<< /Type /Pages /Kids [] /Count 0 >>")

    for page_lines in pages:
        y_start = page_height - margin_top
        content_ops = [b"BT", f"/F1 {font_size} Tf".encode("ascii"), f"{margin_left} {y_start} Td".encode("ascii")]
        first = True
        for ln in page_lines:
            if not first:
                content_ops.append(f"0 {-leading} Td".encode("ascii"))
            first = False
            content_ops.append(f"({pdf_escape_text(ln)}) Tj".encode("latin-1", "replace"))
        content_ops.append(b"ET")
        stream = b"\n".join(content_ops) + b"\n"
        stream_obj = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
        content_obj_id = add_obj(stream_obj)
        content_obj_ids.append(content_obj_id)

        page_obj = (
            b"<< /Type /Page /Parent "
            + f"{pages_obj_id} 0 R".encode("ascii")
            + b" /MediaBox [0 0 "
            + f"{page_width} {page_height}".encode("ascii")
            + b"] /Resources << /Font << /F1 "
            + f"{font_obj_id} 0 R".encode("ascii")
            + b" >> >> /Contents "
            + f"{content_obj_id} 0 R".encode("ascii")
            + b" >>"
        )
        page_obj_id = add_obj(page_obj)
        page_obj_ids.append(page_obj_id)

    kids = b"[ " + b" ".join(f"{pid} 0 R".encode("ascii") for pid in page_obj_ids) + b" ]"
    pages_obj = b"<< /Type /Pages /Kids " + kids + b" /Count " + str(len(page_obj_ids)).encode("ascii") + b" >>"
    objects[pages_obj_id - 1] = pages_obj

    catalog_obj_id = add_obj(b"<< /Type /Catalog /Pages " + f"{pages_obj_id} 0 R".encode("ascii") + b" >>")

    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: List[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(b"xref\n")
    out.extend(f"0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(b"trailer\n")
    out.extend(b"<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root " + f"{catalog_obj_id} 0 R".encode("ascii") + b" >>\n")
    out.extend(b"startxref\n")
    out.extend(f"{xref_start}\n".encode("ascii"))
    out.extend(b"%%EOF\n")

    pdf_path.write_bytes(bytes(out))


def medical_docs_schema() -> Dict[str, Any]:
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "required": ["file_name", "doc_type", "date", "patient_id", "doctor_id", "title", "content"],
            "properties": {
                "file_name": {"type": "STRING"},
                "doc_type": {"type": "STRING"},
                "date": {"type": "STRING"},
                "patient_id": {"type": "STRING"},
                "doctor_id": {"type": "STRING"},
                "title": {"type": "STRING"},
                "content": {"type": "STRING"},
            },
        },
    }


def load_daily_checkins_for_patient(
    db: firestore.Client,
    collection: str,
    patient_id: str,
    start_day: date,
    days: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for day_index in range(days):
        d = start_day + timedelta(days=day_index)
        doc_id = f"{patient_id}_{d.strftime('%Y%m%d')}"
        snap = db.collection(collection).document(doc_id).get()
        if not snap.exists:
            continue
        data = snap.to_dict() or {}
        vitals_arr = data.get("vitals") or []
        vitals = vitals_arr[0] if isinstance(vitals_arr, list) and vitals_arr else {}
        notes = data.get("Notes") or {}
        out.append(
            {
                "doc_id": doc_id,
                "date": str(data.get("Date") or d.isoformat()),
                "day_index": day_index,
                "doctor_id": str(data.get("doctorId") or ""),
                "vitals": {
                    "BP": vitals.get("BP"),
                    "HR": vitals.get("HR"),
                    "RR": vitals.get("RR"),
                    "SpO2": vitals.get("SpO2"),
                    "Temp": vitals.get("Temp"),
                    "Pulse": vitals.get("Pulse"),
                    "heightCm": vitals.get("heightCm"),
                    "weightKg": vitals.get("weightKg"),
                    "BMI": vitals.get("BMI"),
                    "waistHip": vitals.get("waistHip"),
                },
                "note_short": str(notes.get("short") or "").strip(),
            }
        )
    return out


def window(entries: List[Dict[str, Any]], end_index: int, days_back: int) -> List[Dict[str, Any]]:
    start = max(0, end_index - (days_back - 1))
    return entries[start : end_index + 1]


def generate_medical_pdfs_for_patient(
    client: genai.Client,
    model: str,
    patient_id: str,
    entries: List[Dict[str, Any]],
    prescriptions_dir: Path,
    lab_reports_dir: Path,
) -> int:
    prescription_indices = [i for i in range(len(entries)) if entries[i].get("day_index", 0) % 5 == 0][:6]
    lab_indices = [i for i in range(len(entries)) if entries[i].get("day_index", 0) % 10 == 0][:3]

    requested: List[Dict[str, Any]] = []
    for i in prescription_indices:
        e = entries[i]
        dt = str(e.get("date"))
        doctor_id = str(e.get("doctor_id"))
        requested.append(
            {
                "file_name": f"{patient_id}_{dt}_prescription.pdf",
                "doc_type": "prescription",
                "date": dt,
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "title": f"Prescription - {patient_id} - {dt}",
                "context_window_days": 5,
                "context": window(entries, i, 5),
            }
        )
    for i in lab_indices:
        e = entries[i]
        dt = str(e.get("date"))
        doctor_id = str(e.get("doctor_id"))
        requested.append(
            {
                "file_name": f"{patient_id}_{dt}_lab_report.pdf",
                "doc_type": "lab_report",
                "date": dt,
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "title": f"Lab Report - {patient_id} - {dt}",
                "context_window_days": 10,
                "context": window(entries, i, 10),
            }
        )

    prompt = (
        "You are generating realistic clinical documents from daily vitals + brief doctor notes.\n"
        "Output MUST be valid JSON matching the schema. No markdown.\n\n"
        "Rules:\n"
        "- Use only the provided vitals + note_short; do not invent symptoms, diagnoses, imaging, or procedures.\n"
        "- Keep the same patient_id and doctor_id as provided per document.\n"
        "- For prescriptions: include a realistic assessment and medication list with dose/frequency/duration + advice.\n"
        "- For lab reports: include ordered tests, plausible numeric results (with units), brief interpretation, and next steps.\n"
        "- Keep content plain ASCII text, newline separated, ~30-70 lines per document.\n"
        "- No controlled substances.\n\n"
        f"Patient story: {patient_story(patient_id)}\n\n"
        f"Requested documents JSON:\n{json.dumps(requested, ensure_ascii=True)}\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
            response_schema=medical_docs_schema(),
        ),
    )
    docs: List[Dict[str, Any]]
    if getattr(resp, "parsed", None) is not None:
        docs = list(resp.parsed)
    else:
        docs = json.loads(resp.text or "[]")

    by_name = {str(d.get("file_name")): d for d in docs if isinstance(d, dict)}
    written = 0
    for req in requested:
        fname = str(req["file_name"])
        doc = by_name.get(fname) or {}
        doc_type = str(doc.get("doc_type") or req["doc_type"])
        title = str(doc.get("title") or req["title"])
        content = str(doc.get("content") or "")

        if not content.strip():
            context_lines = []
            for c in (req.get("context") or [])[-5:]:
                context_lines.append(f"{c.get('date')} vitals={c.get('vitals')} note={c.get('note_short','')[:120]}")
            content = "\n".join(
                [
                    f"Patient: {req['patient_id']}",
                    f"Doctor: {req['doctor_id']}",
                    f"Date: {req['date']}",
                    "",
                    "Context:",
                    *context_lines,
                    "",
                    "Generated content missing from model response.",
                ]
            )

        out_dir = prescriptions_dir if doc_type == "prescription" else lab_reports_dir
        out_path = out_dir / fname
        write_simple_text_pdf(out_path, title=title, content=content)
        written += 1

    return written


def generate_medical_pdfs(
    db: firestore.Client,
    client: genai.Client,
    collection: str,
    model: str,
    days: int,
    dataset_total_days: int,
    start_day: date,
    prescriptions_dir: Path,
    lab_reports_dir: Path,
) -> int:
    patient_ids, _ = build_seed_ids()
    prescriptions_dir.mkdir(parents=True, exist_ok=True)
    lab_reports_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for pid in patient_ids:
        entries = load_daily_checkins_for_patient(
            db=db,
            collection=collection,
            patient_id=pid,
            start_day=start_day,
            days=days,
        )
        total += generate_medical_pdfs_for_patient(
            client=client,
            model=model,
            patient_id=pid,
            entries=entries,
            prescriptions_dir=prescriptions_dir,
            lab_reports_dir=lab_reports_dir,
        )
    return total


def infer_storage_bucket(service_account_path: str) -> str:
    try:
        data = json.loads(Path(service_account_path).read_text(encoding="utf-8"))
        project_id = str(data.get("project_id") or "").strip()
        if project_id:
            return f"{project_id}.appspot.com"
    except Exception:
        pass
    return "meddflow-dev-8397c.appspot.com"


def normalize_bucket_name(bucket_name: str) -> str:
    b = (bucket_name or "").strip()
    if b.startswith("gs://"):
        b = b[5:]
    return b.strip().strip("/")


def resolve_storage_bucket(bucket_arg: str, service_account_path: str) -> str:
    candidates: List[str] = []
    raw = normalize_bucket_name(bucket_arg)
    if raw:
        candidates.append(raw)
        if raw.endswith(".appspot.com"):
            candidates.append(raw[: -len(".appspot.com")] + ".firebasestorage.app")
        elif raw.endswith(".firebasestorage.app"):
            candidates.append(raw[: -len(".firebasestorage.app")] + ".appspot.com")

    inferred = normalize_bucket_name(infer_storage_bucket(service_account_path))
    if inferred:
        candidates.append(inferred)
        if inferred.endswith(".appspot.com"):
            candidates.append(inferred[: -len(".appspot.com")] + ".firebasestorage.app")
        elif inferred.endswith(".firebasestorage.app"):
            candidates.append(inferred[: -len(".firebasestorage.app")] + ".appspot.com")

    seen: set[str] = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for c in ordered:
        try:
            b = storage.bucket(name=c)
            if b.exists():
                return c
        except Exception:
            continue

    return ordered[0] if ordered else infer_storage_bucket(service_account_path)


def firebase_download_url(bucket_name: str, object_path: str, token: str) -> str:
    encoded = quote(object_path, safe="")
    return f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{encoded}?alt=media&token={token}"


def get_or_set_download_token(blob: Any) -> str:
    try:
        blob.reload()
        md = blob.metadata or {}
        tok = str(md.get("firebaseStorageDownloadTokens") or "").strip()
        if tok:
            return tok.split(",")[0].strip()
    except Exception:
        pass

    tok = str(uuid.uuid4())
    blob.metadata = {**(blob.metadata or {}), "firebaseStorageDownloadTokens": tok}
    try:
        blob.patch()
    except Exception:
        pass
    return tok


def upload_pdfs_and_write_links(
    bucket_name: str,
    src_dir: Path,
    dst_prefix: str,
    out_json_path: Path,
    overwrite: bool,
) -> Dict[str, str]:
    src_dir.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    bucket = storage.bucket(name=bucket_name)
    mapping: Dict[str, str] = {}

    for pdf_path in sorted(src_dir.glob("*.pdf")):
        name = pdf_path.name
        object_path = f"{dst_prefix.rstrip('/')}/{name}"
        blob = bucket.blob(object_path)

        exists = False
        try:
            exists = blob.exists()
        except Exception:
            exists = False

        if exists and not overwrite:
            token = get_or_set_download_token(blob)
            mapping[name] = firebase_download_url(bucket_name, object_path, token)
            continue

        token = str(uuid.uuid4())
        blob.metadata = {**(blob.metadata or {}), "firebaseStorageDownloadTokens": token}
        blob.upload_from_filename(str(pdf_path), content_type="application/pdf")
        mapping[name] = firebase_download_url(bucket_name, object_path, token)

    out_json_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return mapping


def parse_prescription_date_from_filename(file_name: str) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_prescription\.pdf$", file_name)
    return m.group(1) if m else ""


def parse_patient_id_from_prescription_filename(file_name: str) -> str:
    m = re.match(r"^(.*?)_\d{4}-\d{2}-\d{2}_prescription\.pdf$", file_name)
    return m.group(1) if m else ""


def parse_lab_report_date_from_filename(file_name: str) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_lab_report\.pdf$", file_name)
    return m.group(1) if m else ""


def parse_patient_id_from_lab_report_filename(file_name: str) -> str:
    m = re.match(r"^(.*?)_\d{4}-\d{2}-\d{2}_lab_report\.pdf$", file_name)
    return m.group(1) if m else ""


def daily_checkins_doc_id(patient_id: str, date_str: str) -> str:
    try:
        y, m, d = (int(x) for x in date_str.split("-", 2))
        return f"{patient_id}_{y:04d}{m:02d}{d:02d}"
    except Exception:
        return ""


def load_top_level_daily_checkin(db: firestore.Client, patient_id: str, date_str: str) -> Dict[str, Any]:
    doc_id = daily_checkins_doc_id(patient_id, date_str)
    if not doc_id:
        return {}
    snap = db.collection("DailyCheckIns").document(doc_id).get()
    if not snap.exists:
        return {}
    return snap.to_dict() or {}


def clone_patient_doc_from_template(
    db: firestore.Client,
    template_patient_doc_id: str,
    new_patient_doc_id: str,
) -> None:
    src = db.collection("Patients").document(template_patient_doc_id).get()
    template = src.to_dict() if src.exists else {}
    payload = dict(template or {})
    payload["userId"] = new_patient_doc_id
    if "userName" in payload:
        payload["userName"] = new_patient_doc_id
    if "displayName" in payload:
        payload["displayName"] = new_patient_doc_id
    if "createdAt" in payload:
        payload["createdAt"] = iso_utc(datetime.now(timezone.utc))
    db.collection("Patients").document(new_patient_doc_id).set(payload, merge=True)


def seed_patient_daily_checkins_from_top_level(
    db: firestore.Client,
    patient_doc_id: str,
    start_day: date,
    days: int,
    overwrite: bool,
) -> int:
    col = db.collection("Patients").document(patient_doc_id).collection("DailyCheckIns")
    written = 0
    for day_index in range(days):
        dt = (start_day + timedelta(days=day_index)).isoformat()
        src = load_top_level_daily_checkin(db, patient_doc_id, dt)
        if not src:
            continue

        doc_ref = col.document()
        session_id = doc_ref.id
        payload = {
            "createdAt": str(src.get("createdAt") or ""),
            "doctorId": str(src.get("doctorId") or ""),
            "sessionId": session_id,
            "vitals": list(src.get("vitals") or []),
            "Notes": dict(src.get("Notes") or {}),
        }
        if (not overwrite) and doc_ref.get().exists:
            continue
        doc_ref.set(payload, merge=True)
        written += 1
    return written


def extract_checkin_date(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("Date", "date"):
        v = data.get(key)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    vitals = data.get("vitals")
    if isinstance(vitals, list) and vitals and isinstance(vitals[0], dict):
        measured_at = vitals[0].get("measuredAt")
        if isinstance(measured_at, str) and len(measured_at) >= 10:
            return measured_at[:10]
    return ""


def backfill_patient_daily_checkins_notes(
    db: firestore.Client,
    patient_doc_id: str,
    start_day: date,
    days: int,
    overwrite: bool,
) -> int:
    col = db.collection("Patients").document(patient_doc_id).collection("DailyCheckIns")
    start_dt = start_day.isoformat()
    end_dt = (start_day + timedelta(days=days - 1)).isoformat()

    batch = db.batch()
    in_batch = 0
    updated = 0

    for doc in col.stream():
        data = doc.to_dict() or {}
        dt = extract_checkin_date(data)
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        if (not overwrite) and data.get("Notes"):
            continue
        src = load_top_level_daily_checkin(db, patient_doc_id, dt)
        notes = src.get("Notes") or {}
        if not notes:
            continue
        payload = {"Notes": dict(notes), "doctorNotes": firestore.DELETE_FIELD}
        batch.set(doc.reference, payload, merge=True)
        in_batch += 1
        updated += 1
        if in_batch >= 450:
            batch.commit()
            batch = db.batch()
            in_batch = 0

    if in_batch:
        batch.commit()

    return updated


def seed_patient_prescriptions_from_links(
    db: firestore.Client,
    patient_doc_id: str,
    prescription_links: Dict[str, str],
    start_day: date,
    days: int,
    overwrite: bool,
) -> int:
    col = db.collection("Patients").document(patient_doc_id).collection("Prescriptions")
    written = 0

    items = []
    for file_name, url in (prescription_links or {}).items():
        if not (isinstance(file_name, str) and isinstance(url, str) and url.strip()):
            continue
        if parse_patient_id_from_prescription_filename(file_name) != patient_doc_id:
            continue
        dt = parse_prescription_date_from_filename(file_name)
        if not dt:
            continue
        items.append((dt, file_name, url.strip()))

    items.sort(key=lambda x: (x[0], x[1]))
    if not items:
        return 0

    start_dt = start_day.isoformat()
    end_dt = (start_day + timedelta(days=days - 1)).isoformat() if days > 0 else start_dt
    items = [x for x in items if start_dt <= x[0] <= end_dt]

    for dt, file_name, url in items:
        src = load_top_level_daily_checkin(db, patient_doc_id, dt)
        created_at = str(src.get("createdAt") or "")
        doctor_id = str(src.get("doctorId") or "")

        doc_ref = col.document()
        session_id = doc_ref.id
        payload = {
            "createdAt": created_at,
            "doctorId": doctor_id,
            "pdfUrl": url,
            "sessionId": session_id,
            "status": "uploaded",
            "date": dt,
        }
        if (not overwrite) and doc_ref.get().exists:
            continue
        doc_ref.set(payload, merge=True)
        written += 1

    return written


def seed_patient_lab_reports_from_links(
    db: firestore.Client,
    patient_doc_id: str,
    lab_report_links: Dict[str, str],
    start_day: date,
    days: int,
    overwrite: bool,
) -> int:
    col = db.collection("Patients").document(patient_doc_id).collection("LabReports")
    written = 0

    items = []
    for file_name, url in (lab_report_links or {}).items():
        if not (isinstance(file_name, str) and isinstance(url, str) and url.strip()):
            continue
        if parse_patient_id_from_lab_report_filename(file_name) != patient_doc_id:
            continue
        dt = parse_lab_report_date_from_filename(file_name)
        if not dt:
            continue
        items.append((dt, file_name, url.strip()))

    items.sort(key=lambda x: (x[0], x[1]))
    if not items:
        return 0

    start_dt = start_day.isoformat()
    end_dt = (start_day + timedelta(days=days - 1)).isoformat() if days > 0 else start_dt
    items = [x for x in items if start_dt <= x[0] <= end_dt]

    now = iso_utc(datetime.now(timezone.utc))
    for dt, file_name, url in items:
        src = load_top_level_daily_checkin(db, patient_doc_id, dt)
        created_at = str(src.get("createdAt") or now)
        doctor_id = str(src.get("doctorId") or "")

        doc_id = Path(file_name).stem
        doc_ref = col.document(doc_id)
        snap = doc_ref.get()
        current = snap.to_dict() if snap.exists else {}
        if (not overwrite) and str((current or {}).get("fileUrl") or "").strip():
            continue
        session_id = str((current or {}).get("sessionId") or doc_id)
        payload = {
            "doctorId": doctor_id,
            "fileUrl": url,
            "sessionId": session_id,
            "patientId": patient_doc_id,
            "createdAt": created_at,
            "date": dt,
            "fileName": file_name,
            "status": "uploaded",
        }
        doc_ref.set(payload, merge=True)
        written += 1

    return written


def select_default_doctor_id(db: firestore.Client) -> str:
    try:
        docs = list(db.collection("Doctors").limit(1).stream())
        return docs[0].id if docs else ""
    except Exception:
        return ""


def upsert_patient_prescriptions_from_links(
    db: firestore.Client,
    patient_doc_id: str,
    prescription_links: Dict[str, str],
    doctor_id: str,
    overwrite_pdf_url: bool,
) -> int:
    now = iso_utc(datetime.now(timezone.utc))
    col = db.collection("Patients").document(patient_doc_id).collection("Prescriptions")
    existing = list(col.stream())

    items = [(k, v) for k, v in (prescription_links or {}).items() if isinstance(k, str) and isinstance(v, str) and v]
    items.sort(key=lambda kv: (parse_prescription_date_from_filename(kv[0]) or "9999-99-99", kv[0]))

    written = 0

    if existing and items:
        placeholder_docs = [d for d in existing if not str((d.to_dict() or {}).get("pdfUrl") or "").strip()]
        if placeholder_docs:
            doc = placeholder_docs[0]
            name, url = items[0]
            payload: Dict[str, Any] = {
                "pdfUrl": url,
                "doctorId": doctor_id,
                "patientId": patient_doc_id,
                "createdAt": now,
                "Date": parse_prescription_date_from_filename(name),
                "fileName": name,
                "sessionId": str((doc.to_dict() or {}).get("sessionId") or doc.id),
            }
            doc.reference.set(payload, merge=True)
            written += 1
            items = items[1:]

    batch = db.batch()
    in_batch = 0

    for name, url in items:
        doc_id = Path(name).stem
        ref = col.document(doc_id)
        snap = ref.get()
        current = snap.to_dict() if snap.exists else {}
        if (not overwrite_pdf_url) and str((current or {}).get("pdfUrl") or "").strip():
            continue
        payload = {
            "pdfUrl": url,
            "doctorId": doctor_id,
            "patientId": patient_doc_id,
            "createdAt": now,
            "Date": parse_prescription_date_from_filename(name),
            "fileName": name,
            "sessionId": doc_id,
        }
        batch.set(ref, payload, merge=True)
        in_batch += 1
        if in_batch >= 450:
            batch.commit()
            written += in_batch
            batch = db.batch()
            in_batch = 0

    if in_batch:
        batch.commit()
        written += in_batch

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-vitals", action="store_true")
    parser.add_argument("--vitals-collection", default="Daily Checkins")
    parser.add_argument("--vitals-doc", default="default")
    parser.add_argument("--seed-daily-checkins", action="store_true")
    parser.add_argument("--daily-checkins-collection", default="DailyCheckIns")
    parser.add_argument("--generate-doctor-notes", action="store_true")
    parser.add_argument("--notes-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--notes-batch-size", type=int, default=15)
    parser.add_argument("--notes-limit", type=int, default=0)
    parser.add_argument("--overwrite-notes", action="store_true")
    parser.add_argument("--generate-medical-pdfs", action="store_true")
    parser.add_argument("--prescriptions-dir", default="/Users/ayrafraihan/Downloads/PDF vectorizer pipeline/prescripitoons")
    parser.add_argument("--lab-reports-dir", default="/Users/ayrafraihan/Downloads/PDF vectorizer pipeline/lab reports")
    parser.add_argument("--pdf-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--upload-medical-pdfs", action="store_true")
    parser.add_argument("--storage-bucket", default="")
    parser.add_argument("--overwrite-upload", action="store_true")
    parser.add_argument("--out-prescriptions-json", default="prescriptions_storage_links.json")
    parser.add_argument("--out-lab-reports-json", default="lab_reports_storage_links.json")
    parser.add_argument("--seed-patient-prescriptions", action="store_true")
    parser.add_argument("--patient-doc-id", default="")
    parser.add_argument("--prescriptions-links-json", default="prescriptions_storage_links.json")
    parser.add_argument("--lab-reports-links-json", default="lab_reports_storage_links.json")
    parser.add_argument("--default-doctor-id", default="")
    parser.add_argument("--overwrite-patient-prescriptions", action="store_true")
    parser.add_argument("--seed-patient-lab-reports", action="store_true")
    parser.add_argument("--seed-patients-like-template", action="store_true")
    parser.add_argument("--template-patient-doc-id", default="r85Jq9g6QafNikCj5uWcHX9iFSy1")
    parser.add_argument("--backfill-patient-daily-notes", action="store_true")
    parser.add_argument("--dataset-days", type=int, default=200)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--doctor-id", default="")
    parser.add_argument("--prescription-id", default="")
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    args = parser.parse_args()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(args.service_account))
    db = firestore.client()

    if args.init_vitals:
        init_default_vitals(db=db, collection=args.vitals_collection, doc_id=args.vitals_doc)
        print("vitals_initialized", f"{args.vitals_collection}/{args.vitals_doc}")
        return

    if args.seed_daily_checkins:
        days = max(1, int(args.days))
        start_day = parse_start_day(args.start_date, days)

        written = seed_daily_checkins(
            db=db,
            collection=args.daily_checkins_collection,
            days=days,
            start_day=start_day,
        )
        print("daily_checkins_written", written)
        print("daily_checkins_collection", args.daily_checkins_collection)
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if args.generate_doctor_notes:
        project_root = Path(__file__).resolve().parent
        load_env_file(project_root / ".env")

        api_key = require_env("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        days = max(1, int(args.days))
        start_day = parse_start_day(args.start_date, days)

        model = choose_first_working_model(
            client,
            [args.notes_model, "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
        )
        batch_size = max(1, min(25, int(args.notes_batch_size)))
        limit = max(0, int(args.notes_limit))

        written = generate_doctor_notes_for_daily_checkins(
            db=db,
            client=client,
            collection=args.daily_checkins_collection,
            model=model,
            days=days,
            start_day=start_day,
            batch_size=batch_size,
            limit=limit,
            overwrite=bool(args.overwrite_notes),
        )
        print("doctor_notes_written", written)
        print("daily_checkins_collection", args.daily_checkins_collection)
        print("notes_model", model)
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if args.generate_medical_pdfs:
        project_root = Path(__file__).resolve().parent
        load_env_file(project_root / ".env")

        api_key = require_env("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        days = max(1, int(args.days))
        dataset_total_days = max(days, int(args.dataset_days))
        start_day = (
            parse_start_day(args.start_date, days)
            if args.start_date.strip()
            else datetime.now(timezone.utc).date() - timedelta(days=dataset_total_days - 1)
        )

        model = choose_first_working_model(
            client,
            [args.pdf_model, "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"],
        )

        written = generate_medical_pdfs(
            db=db,
            client=client,
            collection=args.daily_checkins_collection,
            model=model,
            days=days,
            dataset_total_days=dataset_total_days,
            start_day=start_day,
            prescriptions_dir=Path(args.prescriptions_dir),
            lab_reports_dir=Path(args.lab_reports_dir),
        )
        print("pdfs_written", written)
        print("prescriptions_dir", args.prescriptions_dir)
        print("lab_reports_dir", args.lab_reports_dir)
        print("pdf_model", model)
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if args.upload_medical_pdfs:
        project_root = Path(__file__).resolve().parent
        bucket_name = resolve_storage_bucket(args.storage_bucket, args.service_account)

        upload_pdfs_and_write_links(
            bucket_name=bucket_name,
            src_dir=Path(args.prescriptions_dir),
            dst_prefix="Prescriptions",
            out_json_path=project_root / args.out_prescriptions_json,
            overwrite=bool(args.overwrite_upload),
        )
        upload_pdfs_and_write_links(
            bucket_name=bucket_name,
            src_dir=Path(args.lab_reports_dir),
            dst_prefix="Lab Reports",
            out_json_path=project_root / args.out_lab_reports_json,
            overwrite=bool(args.overwrite_upload),
        )
        print("storage_bucket", bucket_name)
        print("prescriptions_links_json", str((project_root / args.out_prescriptions_json).resolve()))
        print("lab_reports_links_json", str((project_root / args.out_lab_reports_json).resolve()))
        return

    if args.seed_patient_prescriptions:
        project_root = Path(__file__).resolve().parent
        patient_doc_id = args.patient_doc_id.strip()
        if not patient_doc_id:
            parser.error("--patient-doc-id is required for --seed-patient-prescriptions")

        links_path = Path(args.prescriptions_links_json)
        if not links_path.is_absolute():
            links_path = project_root / links_path
        prescription_links = json.loads(links_path.read_text(encoding="utf-8"))
        if not isinstance(prescription_links, dict):
            raise ValueError("prescriptions-links-json must be a JSON object mapping filename->url")

        doctor_id = args.default_doctor_id.strip() or select_default_doctor_id(db)

        written = upsert_patient_prescriptions_from_links(
            db=db,
            patient_doc_id=patient_doc_id,
            prescription_links=prescription_links,
            doctor_id=doctor_id,
            overwrite_pdf_url=bool(args.overwrite_patient_prescriptions),
        )
        print("patient_prescriptions_written", written)
        print("patient_doc_id", patient_doc_id)
        print("doctor_id", doctor_id)
        print("prescriptions_links_json", str(links_path.resolve()))
        return

    if args.seed_patient_lab_reports:
        project_root = Path(__file__).resolve().parent
        patient_doc_id = args.patient_doc_id.strip()
        if not patient_doc_id:
            parser.error("--patient-doc-id is required for --seed-patient-lab-reports")

        links_path = Path(args.lab_reports_links_json)
        if not links_path.is_absolute():
            links_path = project_root / links_path
        lab_links = json.loads(links_path.read_text(encoding="utf-8"))
        if not isinstance(lab_links, dict):
            raise ValueError("lab-reports-links-json must be a JSON object mapping filename->url")

        days = max(1, int(args.days))
        start_day = parse_start_day(args.start_date, days)

        written = seed_patient_lab_reports_from_links(
            db=db,
            patient_doc_id=patient_doc_id,
            lab_report_links=lab_links,
            start_day=start_day,
            days=days,
            overwrite=bool(args.overwrite_patient_prescriptions),
        )
        print("patient_lab_reports_written", written)
        print("patient_doc_id", patient_doc_id)
        print("lab_reports_links_json", str(links_path.resolve()))
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if args.seed_patients_like_template:
        project_root = Path(__file__).resolve().parent
        links_path = Path(args.prescriptions_links_json)
        if not links_path.is_absolute():
            links_path = project_root / links_path
        prescription_links = json.loads(links_path.read_text(encoding="utf-8"))
        if not isinstance(prescription_links, dict):
            raise ValueError("prescriptions-links-json must be a JSON object mapping filename->url")

        patient_ids = sorted(
            {pid for pid in (parse_patient_id_from_prescription_filename(k) for k in prescription_links.keys()) if pid}
        )
        if not patient_ids:
            raise ValueError("No patient ids found in prescriptions-links-json")

        days = max(1, int(args.days))
        if args.start_date.strip():
            start_day = parse_start_day(args.start_date, days)
        else:
            all_dates = sorted(
                {parse_prescription_date_from_filename(k) for k in prescription_links.keys() if parse_prescription_date_from_filename(k)}
            )
            if all_dates:
                y, m, d = (int(x) for x in all_dates[0].split("-", 2))
                start_day = date(y, m, d)
            else:
                start_day = datetime.now(timezone.utc).date() - timedelta(days=days - 1)

        total_patients = 0
        total_daily = 0
        lab_links_path = Path(args.lab_reports_links_json)
        if not lab_links_path.is_absolute():
            lab_links_path = project_root / lab_links_path
        lab_links = json.loads(lab_links_path.read_text(encoding="utf-8")) if lab_links_path.exists() else {}
        if lab_links and not isinstance(lab_links, dict):
            raise ValueError("lab-reports-links-json must be a JSON object mapping filename->url")

        total_prescriptions = 0
        total_lab_reports = 0

        for pid in patient_ids:
            clone_patient_doc_from_template(
                db=db,
                template_patient_doc_id=str(args.template_patient_doc_id).strip(),
                new_patient_doc_id=pid,
            )
            total_patients += 1
            total_daily += seed_patient_daily_checkins_from_top_level(
                db=db,
                patient_doc_id=pid,
                start_day=start_day,
                days=days,
                overwrite=bool(args.overwrite_patient_prescriptions),
            )
            total_prescriptions += seed_patient_prescriptions_from_links(
                db=db,
                patient_doc_id=pid,
                prescription_links=prescription_links,
                start_day=start_day,
                days=days,
                overwrite=bool(args.overwrite_patient_prescriptions),
            )
            if lab_links:
                total_lab_reports += seed_patient_lab_reports_from_links(
                    db=db,
                    patient_doc_id=pid,
                    lab_report_links=lab_links,
                    start_day=start_day,
                    days=days,
                    overwrite=bool(args.overwrite_patient_prescriptions),
                )

        print("patients_written", total_patients)
        print("patient_daily_checkins_written", total_daily)
        print("patient_prescriptions_written", total_prescriptions)
        if lab_links:
            print("patient_lab_reports_written", total_lab_reports)
        print("template_patient_doc_id", str(args.template_patient_doc_id).strip())
        print("prescriptions_links_json", str(links_path.resolve()))
        if lab_links:
            print("lab_reports_links_json", str(lab_links_path.resolve()))
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if args.backfill_patient_daily_notes:
        project_root = Path(__file__).resolve().parent
        links_path = Path(args.prescriptions_links_json)
        if not links_path.is_absolute():
            links_path = project_root / links_path
        prescription_links = json.loads(links_path.read_text(encoding="utf-8"))
        if not isinstance(prescription_links, dict):
            raise ValueError("prescriptions-links-json must be a JSON object mapping filename->url")

        patient_ids = sorted(
            {pid for pid in (parse_patient_id_from_prescription_filename(k) for k in prescription_links.keys()) if pid}
        )
        if not patient_ids:
            raise ValueError("No patient ids found in prescriptions-links-json")

        days = max(1, int(args.days))
        start_day = parse_start_day(args.start_date, days)

        total_updated = 0
        for pid in patient_ids:
            total_updated += backfill_patient_daily_checkins_notes(
                db=db,
                patient_doc_id=pid,
                start_day=start_day,
                days=days,
                overwrite=bool(args.overwrite_patient_prescriptions),
            )

        print("patient_daily_notes_updated", total_updated)
        print("patients", len(patient_ids))
        print("prescriptions_links_json", str(links_path.resolve()))
        print("date_range", start_day.isoformat(), (start_day + timedelta(days=days - 1)).isoformat())
        return

    if not args.doctor_id.strip() or not args.prescription_id.strip():
        parser.error("--doctor-id and --prescription-id are required unless --init-vitals is set")

    pres_ref = (
        db.collection("Doctors")
        .document(args.doctor_id)
        .collection("Prescriptions")
        .document(args.prescription_id)
    )

    pres_snap = pres_ref.get()

    print("prescription_doc_exists", pres_snap.exists)
    if pres_snap.exists:
        vz = (pres_snap.to_dict() or {}).get("vectorization") or {}
        print("prescription.vectorization.status", vz.get("status"))
        print("prescription.vectorization.model", vz.get("model"))
        print("prescription.vectorization.dimension", vz.get("dimension"))
        run_id = vz.get("latest_run_id")
        print("prescription.vectorization.latest_run_id", run_id)
        vectors = list(pres_ref.collection("Vectors").select(["chunk_id"]).stream())
        print("prescription.vector_docs", len(vectors))


if __name__ == "__main__":
    main()
