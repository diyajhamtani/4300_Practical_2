import chromadb
import ollama
import numpy as np
import os
import fitz
from sentence_transformers import SentenceTransformer
import re

# Initialize ChromaDB client
db = chromadb.PersistentClient(os.path.join(".", "chroma_db"))
collection = db.get_or_create_collection(name=(os.getenv("COLLECTION_NAME", "embedding_index")))
PREPROCESSING = os.getenv("PREPROCESSING", True)

VECTOR_DIM = int(os.getenv("VECTOR_DIM", 384))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))

# Get embedding model from environment variable
CURRENT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Using model: {CURRENT_MODEL}")

def clear_chroma_store():
    print("Clearing existing ChromaDB store...")
    global collection
    collection_ids = collection.get(ids=None)["ids"]
    if collection_ids:
        collection.delete(ids=collection_ids)
    print("ChromaDB store cleared.")

def get_embedding(text: str, model: str = CURRENT_MODEL) -> list:
    if "sentence-transformers" in model or "instructor" in model:
        transformer = SentenceTransformer(model)
        return transformer.encode(text).tolist()
    else:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk[:30]}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    print(f"Stored embedding for: {chunk[:30]}...")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = [(page_num, page.get_text()) for page_num, page in enumerate(doc)]
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def preprocess_text(text):
    # Remove special characters (anything that is not a letter, number, or space)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text

def process_pdfs(data_dir):
    """Processes all PDFs in the data directory and stores embeddings in Redis."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                # Preprocess text here before chunking
                if PREPROCESSING:
                    text = preprocess_text(text)
                chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE)

                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                    )

            print(f"Processed {file_name}")

def query_chroma(query_text: str, k=5):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    
    for doc, score in zip(results["metadatas"][0], results["distances"][0]):
        print(f"{doc['file']} (Page {doc['page']})\n ----> Score: {score}\n")

def main():
    clear_chroma_store()
    process_pdfs(os.path.join("data"))
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")

if __name__ == "__main__":
    main()
