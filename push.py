import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
import firebase_admin
from firebase_admin import credentials, firestore

from main import PDFRAGSystem, load_env_file, require_env


def get_pdf_url(db: firestore.Client, doctor_id: str, prescription_id: str) -> str:
    doc = (
        db.collection("Doctors")
        .document(doctor_id)
        .collection("Prescriptions")
        .document(prescription_id)
        .get()
    )

    if not doc.exists:
        raise ValueError("Prescription not found")

    data = doc.to_dict() or {}
    pdf_url = data.get("pdfUrl")
    if not pdf_url:
        raise ValueError("Missing pdfUrl on prescription document")
    return str(pdf_url)


def download_pdf_to_temp(pdf_url: str) -> Path:
    r = requests.get(pdf_url, timeout=60)
    r.raise_for_status()

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    tmp = Path(path)
    tmp.write_bytes(r.content)
    return tmp


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def push_vectors(
    db: firestore.Client,
    doctor_id: str,
    prescription_id: str,
    result: Dict[str, Any],
    run_id: str,
) -> None:
    prescription_ref = (
        db.collection("Doctors")
        .document(doctor_id)
        .collection("Prescriptions")
        .document(prescription_id)
    )

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
        "error": metadata.get("embedding_error"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_run_id": run_id,
    }

    vectors = result.get("vectors") or []
    if not isinstance(vectors, list):
        raise ValueError("Invalid vectors payload")

    vector_docs = []
    for v in vectors:
        if not isinstance(v, dict):
            continue
        chunk_id = str(v.get("chunk_id", ""))
        if not chunk_id:
            continue
        vector_docs.append((chunk_id, v))

    prescription_ref.set({"vectorization": summary}, merge=True)

    for group in iter_batches(vector_docs, 500):
        batch = db.batch()
        for chunk_id, v in group:
            vec_ref = prescription_ref.collection("Vectors").document(chunk_id)
            payload = {
                "chunk_id": v.get("chunk_id"),
                "page_index": v.get("page_index"),
                "start_char": v.get("start_char"),
                "end_char": v.get("end_char"),
                "chunk_length": v.get("chunk_length"),
                "chunk_text": v.get("chunk_text"),
                "embedding": v.get("embedding"),
                "embedding_dimension": v.get("embedding_dimension"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
            }
            batch.set(vec_ref, payload, merge=True)
        batch.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor-id", required=True)
    parser.add_argument("--prescription-id", required=True)
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pdf-url", default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / ".env")

    if not firebase_admin._apps:
        cred = credentials.Certificate(args.service_account)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    pdf_url = args.pdf_url.strip() or get_pdf_url(db, args.doctor_id, args.prescription_id)
    tmp_pdf = download_pdf_to_temp(pdf_url)
    try:
        rag = PDFRAGSystem(
            mistral_api_key=require_env("MISTRAL_API_KEY"),
            google_api_key=require_env("GOOGLE_API_KEY"),
        )
        result = rag.process_pdf(
            pdf_path=str(tmp_pdf),
            user_id=args.doctor_id,
            session_id=args.prescription_id,
        )
    finally:
        try:
            tmp_pdf.unlink(missing_ok=True)
        except Exception:
            pass

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    push_vectors(
        db=db,
        doctor_id=args.doctor_id,
        prescription_id=args.prescription_id,
        result=result,
        run_id=run_id,
    )

    print("Pushed vectors to Firestore")


if __name__ == "__main__":
    main()
