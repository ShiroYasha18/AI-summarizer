import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from google.genai import types

from main import load_env_file, require_env


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / norm for x in vec]


def dot(a: List[float], b: List[float]) -> float:
    return float(sum((x * y) for x, y in zip(a, b)))


def embed_query(client: genai.Client, text: str, output_dim: int) -> List[float]:
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=output_dim),
    )
    vec = res.embeddings[0].values
    return l2_normalize(vec)


def load_vectors_from_firestore(
    db: firestore.Client,
    doctor_id: str,
    prescription_id: str,
) -> List[Dict[str, Any]]:
    pres_ref = (
        db.collection("Doctors")
        .document(doctor_id)
        .collection("Prescriptions")
        .document(prescription_id)
    )
    docs = pres_ref.collection("Vectors").select(
        [
            "chunk_id",
            "chunk_text",
            "page_index",
            "embedding",
            "embedding_dimension",
        ]
    ).stream()
    out: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        if "chunk_id" not in data:
            data["chunk_id"] = d.id
        out.append(data)
    return out


def load_vectors_from_json(path: Path) -> List[Dict[str, Any]]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    vectors = data.get("vectors") or []
    if not isinstance(vectors, list):
        raise ValueError("Invalid JSON vectors")
    return [v for v in vectors if isinstance(v, dict)]


def rank_vectors(query_vec: List[float], vectors: List[Dict[str, Any]], top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
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
    return scored[:top_k]


def answer_with_context(client: genai.Client, query: str, contexts: List[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    prompt = (
        "Answer the question using only the provided context. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion:\n{query}"
    )
    model = "gemini-2.0-flash"
    res = client.models.generate_content(model=model, contents=prompt)
    return str(getattr(res, "text", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--doctor-id", default="Sx0kJEsmrQYQHZqJbWPWqutz8Sa2")
    parser.add_argument("--prescription-id", default="An0kJEs3rQYQAZqJbEPWqutz8Sn2")
    parser.add_argument("--service-account", default="serviceAccountKey.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--from-json", default="")
    parser.add_argument("--no-answer", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / ".env")

    google_key = require_env("GOOGLE_API_KEY")
    client = genai.Client(api_key=google_key)

    query_vec = embed_query(client, args.query, output_dim=1536)

    vectors: List[Dict[str, Any]]
    if args.from_json.strip():
        vectors = load_vectors_from_json(Path(args.from_json))
    else:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(args.service_account))
        db = firestore.client()
        vectors = load_vectors_from_firestore(db, args.doctor_id, args.prescription_id)

    ranked = rank_vectors(query_vec, vectors, top_k=args.top_k)

    print("top_matches")
    contexts: List[str] = []
    for score, v in ranked:
        chunk_id = v.get("chunk_id")
        page_index = v.get("page_index")
        text = str(v.get("chunk_text", "")).strip()
        contexts.append(text)
        preview = text[:240].replace("\n", " ").strip()
        print(f"- score={score:.4f} chunk_id={chunk_id} page_index={page_index} text_preview={preview}")

    if not args.no_answer:
        print("\nanswer")
        print(answer_with_context(client, args.query, contexts))


if __name__ == "__main__":
    main()

