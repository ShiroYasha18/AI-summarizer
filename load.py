import requests
import argparse
import tempfile
from pathlib import Path
from pypdf import PdfReader
import firebase_admin
from firebase_admin import credentials, firestore

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


def download_pdf(pdf_url: str, path: str | None = None) -> str:
    r = requests.get(pdf_url, timeout=60)
    r.raise_for_status()

    if path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        Path(tmp_path).write_bytes(r.content)
        return tmp_path

    Path(path).write_bytes(r.content)
    return path


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor-id", required=True)
    parser.add_argument("--prescription-id", required=True)
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--out-pdf", default="temp.pdf")
    parser.add_argument("--out-text", default="document.txt")
    args = parser.parse_args()

    if not firebase_admin._apps:
        cred = credentials.Certificate(args.service_account)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    pdf_url = get_pdf_url(db, args.doctor_id, args.prescription_id)
    pdf_path = download_pdf(pdf_url, args.out_pdf)

    text = extract_text(pdf_path)
    Path(args.out_text).write_text(text, encoding="utf-8")
    print(f"Saved PDF to {args.out_pdf} and text to {args.out_text}")
