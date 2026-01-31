import os
import json
import base64
import hashlib
import math
from typing import List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from mistralai import Mistral
from google import genai
from google.genai import types

class PDFRAGSystem:
    """
    A RAG system that processes PDFs using Mistral Document AI and Gemini embeddings.
    
    Flow:
    1. OCR PDF using Mistral Document AI
    2. Chunk the document with semantic overlap
    3. Generate embeddings using Gemini Embedding
    4. Store vectors with metadata in JSON format
    """
    
    def __init__(
        self,
        mistral_api_key: str,
        google_api_key: str,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
        embedding_model: str = "gemini-embedding-001",
        embedding_output_dim: int = 1536,
        embedding_batch_size: int = 64,
    ):
        """Initialize the RAG system with API clients."""
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.gemini_client = genai.Client(api_key=google_api_key)
        
        # Configuration
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.embedding_model = str(embedding_model)
        self.embedding_output_dim = int(embedding_output_dim)
        self.embedding_batch_size = max(1, int(embedding_batch_size))

    def parse_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        ocr_response = self.mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
        )

        pages = getattr(ocr_response, "pages", None)
        if not pages:
            raise ValueError("No pages returned from Mistral OCR")

        extracted_pages: List[Dict[str, Any]] = []
        for p in pages:
            markdown = getattr(p, "markdown", None)
            if not markdown:
                continue
            extracted_pages.append(
                {
                    "page_index": int(getattr(p, "index", len(extracted_pages))),
                    "text": markdown,
                    "text_length": len(markdown),
                }
            )

        if not extracted_pages:
            raise ValueError("Empty OCR text returned from Mistral OCR")

        return extracted_pages
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text with semantic overlap.
        Returns list of chunks with metadata.
        """
        chunks = []
        chunk_id = 0
        
        # Simple character-based chunking with overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start, end - 200)
                search_end = min(len(text), end + 200)
                
                # Find last sentence ending in the search window
                for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_pos = text.rfind(delimiter, search_start, search_end)
                    if last_pos != -1:
                        end = last_pos + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "chunk_text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "chunk_length": len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
        return chunks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in pages:
            page_index = int(page["page_index"])
            page_text = str(page["text"])
            page_chunks = self.chunk_text(page_text)
            for i, c in enumerate(page_chunks):
                chunks.append(
                    {
                        **c,
                        "chunk_id": f"p{page_index}_chunk_{i}",
                        "page_index": page_index,
                    }
                )
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks using Gemini."""
        texts = [chunk["chunk_text"] for chunk in chunks]

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch_texts = texts[i : i + self.embedding_batch_size]
            result = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=batch_texts,
                config=types.EmbedContentConfig(output_dimensionality=self.embedding_output_dim),
            )
            embeddings.extend([e.values for e in result.embeddings])
        if len(embeddings) != len(chunks):
            raise ValueError(f"Gemini embeddings count mismatch: {len(embeddings)} vs {len(chunks)}")

        for i, chunk in enumerate(chunks):
            vec = embeddings[i]
            norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            chunk["embedding"] = vec
            chunk["embedding_dimension"] = len(vec)

        return chunks
    
    def process_pdf(self, pdf_path: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Main processing pipeline for PDF.
        
        Returns JSON with all vector details and metadata.
        """
        print(f"Starting PDF processing: {pdf_path}")
        
        # Step 1: OCR PDF
        pdf_link_id = ""
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
            pdf_link_id = f"sha256:{hashlib.sha256(pdf_bytes).hexdigest()}"
            print("Parsing PDF with Mistral OCR...")
            extracted_pages = self.parse_pdf(pdf_bytes)
        except Exception as e:
            print(f"OCR processing failed: {e}")
            return {
                "user_id": user_id,
                "session_id": session_id,
                "pdf_link_id": pdf_link_id,
                "date_uploaded": datetime.now(timezone.utc).isoformat(),
                "date_vectorized": datetime.now(timezone.utc).isoformat(),
                "vectorization_status": "uploaded, ocr_failed",
                "metadata": {
                    "source_file": os.path.basename(pdf_path),
                    "error": str(e),
                },
                "pages": [],
                "vectors": []
            }
        
        # Step 2: Chunk text
        print("Chunking document...")
        chunks = self.chunk_pages(extracted_pages)
        print(f"Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("Generating embeddings with Gemini...")
        vectorization_status = "uploaded, ocr, chunked, vectorized"
        embedding_provider = "gemini"
        embedding_model = self.embedding_model
        embedding_error: str | None = None
        try:
            vectorized_chunks = self.generate_embeddings(chunks)
        except Exception as e:
            embedding_error = f"{type(e).__name__}: {e}"
            print(f"Gemini embedding failed: {embedding_error}")
            vectorized_chunks = chunks
            vectorization_status = "uploaded, ocr, chunked, embedding_failed"
        
        # Step 4: Create final JSON structure
        embedding_dimension = 0
        if vectorized_chunks:
            embedding_dimension = int(vectorized_chunks[0].get("embedding_dimension", 0))

        result = {
            "user_id": user_id,
            "session_id": session_id,
            "pdf_link_id": pdf_link_id,
            "date_uploaded": datetime.now(timezone.utc).isoformat(),
            "date_vectorized": datetime.now(timezone.utc).isoformat(),
            "vectorization_status": vectorization_status,
            "metadata": {
                "source_file": os.path.basename(pdf_path),
                "total_pages": len(extracted_pages),
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_output_dim": self.embedding_output_dim,
                "embedding_dimension": embedding_dimension,
                "embedding_error": embedding_error,
            },
            "pages": extracted_pages,
            "vectors": vectorized_chunks
        }
        
        print("Processing complete!")
        return result
    
    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """Save the vectorized data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def run_batch(pdf_dir: Path, output_dir: Path, user_id: str) -> None:
    pdf_paths = sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not pdf_paths:
        print(f"No PDFs found in: {pdf_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    rag_system = PDFRAGSystem(
        mistral_api_key=require_env("MISTRAL_API_KEY"),
        google_api_key=require_env("GOOGLE_API_KEY"),
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for pdf_path in pdf_paths:
        session_id = f"{pdf_path.stem}_{run_id}"
        result = rag_system.process_pdf(
            pdf_path=str(pdf_path),
            user_id=user_id,
            session_id=session_id,
        )
        output_path = output_dir / f"{pdf_path.stem}.json"
        rag_system.save_to_json(result, str(output_path))


# Example usage
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / ".env")

    pdf_dir = project_root / "pdfs"
    output_dir = project_root / "vectorized_outputs"
    run_batch(pdf_dir=pdf_dir, output_dir=output_dir, user_id="local_user")
