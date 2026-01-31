import argparse

import firebase_admin
from firebase_admin import credentials, firestore


def delete_collection(db: firestore.Client, col: firestore.CollectionReference, batch_size: int = 500) -> int:
    deleted = 0
    while True:
        docs = list(col.limit(batch_size).stream())
        if not docs:
            break
        batch = db.batch()
        for d in docs:
            batch.delete(d.reference)
        batch.commit()
        deleted += len(docs)
    return deleted


def delete_vector_runs(db: firestore.Client, doctor_id: str, prescription_id: str) -> int:
    pres_ref = (
        db.collection("Doctors")
        .document(doctor_id)
        .collection("Prescriptions")
        .document(prescription_id)
    )
    runs_col = pres_ref.collection("VectorRuns")

    total_deleted = 0
    run_docs = list(runs_col.stream())
    for run_doc in run_docs:
        vectors_col = run_doc.reference.collection("Vectors")
        total_deleted += delete_collection(db, vectors_col)
        run_doc.reference.delete()
        total_deleted += 1
    return total_deleted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor-id", required=True)
    parser.add_argument("--prescription-id", required=True)
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--confirm", action="store_true", default=False)
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("Refusing to delete without --confirm")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(args.service_account))
    db = firestore.client()

    deleted = delete_vector_runs(db, args.doctor_id, args.prescription_id)
    print(f"deleted_docs={deleted}")


if __name__ == "__main__":
    main()

