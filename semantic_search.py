import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

UPLOAD_FOLDER = "uploads"
CHUNK_SIZE = 500  # characters

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper to extract text from PDFs
def extract_text_from_pdfs(folder):
    chunks, meta = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            doc = fitz.open(path)
            full_text = " ".join(page.get_text() for page in doc)
            for i in range(0, len(full_text), CHUNK_SIZE):
                chunk = full_text[i:i+CHUNK_SIZE]
                chunks.append(chunk)
                meta.append({"filename": filename, "chunk_index": i})
    return chunks, meta

# Embed all chunks
def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=True)

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Search function
def search(query, index, chunks, meta, top_k=5):
    q_vec = model.encode([query])
    D, I = index.search(q_vec, top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        results.append({
            "text": chunks[i],
            "filename": meta[i]["filename"],
            "score": float(score)
        })
    return results

# Example usage
if __name__ == "__main__":
    print("📥 Loading PDFs from 'uploads/'...")
    chunks, meta = extract_text_from_pdfs(UPLOAD_FOLDER)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    while True:
        query = input("\n🔍 Ask a question: ")
        if not query:
            break
        results = search(query, index, chunks, meta)
        for r in results:
            print(f"\n📄 {r['filename']}\nScore: {r['score']:.2f}\n---\n{r['text'][:500]}...")
