import argparse
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import firebase_admin
import requests
from firebase_admin import credentials, firestore

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import PDFRAGSystem, load_env_file, require_env


logger = logging.getLogger("vectorizer")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def download_pdf_to_temp(pdf_url: str) -> Path:
    start = time.perf_counter()
    r = requests.get(pdf_url, timeout=90)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    tmp = Path(path)
    tmp.write_bytes(r.content)
    logger.info(
        "pdf_downloaded bytes=%s duration_ms=%s",
        len(r.content),
        int((time.perf_counter() - start) * 1000),
    )
    return tmp


def normalize_status(v: Any) -> str:
    return str(v or "").strip().lower()


def is_already_vectorized(doc_data: Dict[str, Any]) -> bool:
    vz = doc_data.get("vectorization") or {}
    if not isinstance(vz, dict):
        return False
    status = normalize_status(vz.get("status"))
    return "vectorized" in status and not normalize_status(vz.get("error"))


def safe_str(v: Any) -> str:
    return str(v) if v is not None else ""


def extract_filename_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        name = Path(path).name
        return name
    except Exception:
        return ""


def push_vectors_to_parent_doc(
    db: firestore.Client,
    doc_ref: firestore.DocumentReference,
    result: Dict[str, Any],
    run_id: str,
    meta: Dict[str, Any],
) -> int:
    now = iso_utc_now()
    metadata = result.get("metadata") or {}
    summary: Dict[str, Any] = {
        "status": result.get("vectorization_status"),
        "pdf_link_id": result.get("pdf_link_id"),
        "date_vectorized": result.get("date_vectorized"),
        "provider": metadata.get("embedding_provider"),
        "model": metadata.get("embedding_model"),
        "dimension": metadata.get("embedding_dimension"),
        "output_dim": metadata.get("embedding_output_dim"),
        "total_pages": metadata.get("total_pages"),
        "total_chunks": metadata.get("total_chunks"),
        "error": metadata.get("embedding_error") or metadata.get("error"),
        "updated_at": now,
        "latest_run_id": run_id,
        "chunk_size": metadata.get("chunk_size"),
        "chunk_overlap": metadata.get("chunk_overlap"),
        **meta,
    }

    doc_ref.set({"vectorization": summary}, merge=True)

    vectors = result.get("vectors") or []
    if not isinstance(vectors, list):
        return 0
    if "vectorized" not in normalize_status(result.get("vectorization_status")):
        return 0

    vector_docs: List[Tuple[str, Dict[str, Any]]] = []
    for v in vectors:
        if not isinstance(v, dict):
            continue
        chunk_id = safe_str(v.get("chunk_id")).strip()
        if not chunk_id:
            continue
        vector_docs.append((chunk_id, v))

    if not vector_docs:
        return 0

    written = 0
    for group in iter_batches(vector_docs, 450):
        batch = db.batch()
        for chunk_id, v in group:
            vec_ref = doc_ref.collection("Vectors").document(chunk_id)
            payload = {
                "chunk_id": v.get("chunk_id"),
                "page_index": v.get("page_index"),
                "start_char": v.get("start_char"),
                "end_char": v.get("end_char"),
                "chunk_length": v.get("chunk_length"),
                "chunk_text": v.get("chunk_text"),
                "embedding": v.get("embedding"),
                "embedding_dimension": v.get("embedding_dimension"),
                "created_at": now,
                "run_id": run_id,
                "document_path": doc_ref.path,
                **meta,
            }
            batch.set(vec_ref, payload, merge=True)
        batch.commit()
        written += len(group)
    return written


def gather_targets(
    db: firestore.Client,
    limit: int,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    scanned = {"patients": 0, "prescriptions": 0, "lab_reports": 0}
    eligible = {"prescriptions": 0, "lab_reports": 0}
    for patient_snap in db.collection("Patients").stream():
        scanned["patients"] += 1
        patient_id = patient_snap.id
        patient_ref = patient_snap.reference

        for pres_snap in patient_ref.collection("Prescriptions").stream():
            scanned["prescriptions"] += 1
            data = pres_snap.to_dict() or {}
            if (not overwrite) and is_already_vectorized(data):
                continue
            if normalize_status(data.get("status")) not in ("", "uploaded"):
                continue
            url = safe_str(data.get("pdfUrl")).strip()
            if not url:
                continue
            eligible["prescriptions"] += 1
            targets.append(
                {
                    "type": "prescription",
                    "ref": pres_snap.reference,
                    "url": url,
                    "meta": {
                        "patientId": patient_id,
                        "doctorId": safe_str(data.get("doctorId")).strip(),
                        "date": safe_str(data.get("date") or data.get("Date")).strip()[:10],
                        "fileName": safe_str(data.get("fileName")).strip() or extract_filename_from_url(url),
                        "sessionId": safe_str(data.get("sessionId")).strip(),
                        "documentType": "prescription",
                    },
                }
            )

        for lab_snap in patient_ref.collection("LabReports").stream():
            scanned["lab_reports"] += 1
            data = lab_snap.to_dict() or {}
            if (not overwrite) and is_already_vectorized(data):
                continue
            if normalize_status(data.get("status")) not in ("", "uploaded"):
                continue
            url = safe_str(data.get("fileUrl")).strip()
            if not url:
                continue
            eligible["lab_reports"] += 1
            targets.append(
                {
                    "type": "lab_report",
                    "ref": lab_snap.reference,
                    "url": url,
                    "meta": {
                        "patientId": patient_id,
                        "doctorId": safe_str(data.get("doctorId")).strip(),
                        "date": safe_str(data.get("date") or data.get("Date")).strip()[:10],
                        "fileName": safe_str(data.get("fileName")).strip() or extract_filename_from_url(url),
                        "sessionId": safe_str(data.get("sessionId")).strip(),
                        "documentType": "lab_report",
                    },
                }
            )

    def sort_key(t: Dict[str, Any]) -> Tuple[str, str, str, str]:
        meta = t.get("meta") or {}
        return (
            safe_str(t.get("type")),
            safe_str(meta.get("patientId")),
            safe_str(meta.get("date")),
            safe_str(meta.get("fileName")),
        )

    targets.sort(key=sort_key)
    logger.info(
        "targets_scanned patients=%s prescriptions=%s lab_reports=%s eligible_prescriptions=%s eligible_lab_reports=%s",
        scanned["patients"],
        scanned["prescriptions"],
        scanned["lab_reports"],
        eligible["prescriptions"],
        eligible["lab_reports"],
    )
    if limit > 0:
        targets = targets[:limit]
    logger.info("targets_selected count=%s limit=%s overwrite=%s", len(targets), limit, overwrite)
    return targets


def vectorize_targets(
    db: firestore.Client,
    rag: PDFRAGSystem,
    targets: List[Dict[str, Any]],
    run_id: str,
) -> Dict[str, int]:
    processed = 0
    vectors_written = 0
    failed = 0

    total = len(targets)
    for idx, t in enumerate(targets, start=1):
        doc_ref: firestore.DocumentReference = t["ref"]
        url = str(t["url"])
        meta = dict(t.get("meta") or {})
        processed += 1
        start = time.perf_counter()
        logger.info(
            "vectorize_start index=%s total=%s type=%s patientId=%s date=%s fileName=%s doc_path=%s",
            idx,
            total,
            safe_str(t.get("type")),
            safe_str(meta.get("patientId")),
            safe_str(meta.get("date")),
            safe_str(meta.get("fileName")),
            doc_ref.path,
        )

        tmp_pdf: Optional[Path] = None
        try:
            tmp_pdf = download_pdf_to_temp(url)
            result = rag.process_pdf(
                pdf_path=str(tmp_pdf),
                user_id=str(meta.get("patientId") or ""),
                session_id=doc_ref.id,
            )
        except Exception as e:
            failed += 1
            logger.exception("vectorize_failed doc_path=%s", doc_ref.path)
            result = {
                "user_id": str(meta.get("patientId") or ""),
                "session_id": doc_ref.id,
                "pdf_link_id": "",
                "date_uploaded": iso_utc_now(),
                "date_vectorized": iso_utc_now(),
                "vectorization_status": "uploaded, vectorize_failed",
                "metadata": {"error": f"{type(e).__name__}: {e}"},
                "pages": [],
                "vectors": [],
            }
        finally:
            if tmp_pdf:
                try:
                    tmp_pdf.unlink(missing_ok=True)
                except Exception:
                    pass

        written = push_vectors_to_parent_doc(db=db, doc_ref=doc_ref, result=result, run_id=run_id, meta=meta)
        vectors_written += written
        logger.info(
            "vectorize_done index=%s total=%s doc_path=%s chunks_written=%s duration_ms=%s status=%s",
            idx,
            total,
            doc_ref.path,
            written,
            int((time.perf_counter() - start) * 1000),
            safe_str(result.get("vectorization_status")),
        )

    return {"processed": processed, "vectors_written": vectors_written, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--run-id", default="")
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

    mistral_key = require_env("MISTRAL_API_KEY")
    google_key = require_env("GOOGLE_API_KEY")
    rag = PDFRAGSystem(
        mistral_api_key=mistral_key,
        google_api_key=google_key,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_batch_size=64,
    )

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info(
        "run_start run_id=%s limit=%s overwrite=%s chunk_size=%s chunk_overlap=%s embedding_model=%s embedding_output_dim=%s",
        run_id,
        int(args.limit),
        bool(args.overwrite),
        rag.chunk_size,
        rag.chunk_overlap,
        rag.embedding_model,
        rag.embedding_output_dim,
    )
    targets = gather_targets(db=db, limit=int(args.limit), overwrite=bool(args.overwrite))
    stats = vectorize_targets(db=db, rag=rag, targets=targets, run_id=run_id)
    logger.info("run_done stats=%s", stats)


if __name__ == "__main__":
    main()
