import chromadb
import ollama
import numpy as np
import os
import fitz
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768

# Available embedding models
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}

# Get embedding model from environment variable
CURRENT_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODELS["minilm"])
print(f"Using model: {CURRENT_MODEL}")

def clear_chroma_store():
    print("Clearing existing ChromaDB store...")
    db.delete_collection("embedding_index")
    global collection
    collection = db.get_or_create_collection(name="embedding_index")
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

def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def query_chroma(query_text: str, k=5):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    
    for doc, score in zip(results["metadatas"][0], results["distances"][0]):
        print(f"{doc['file']} (Page {doc['page']})\n ----> Score: {score}\n")

def main():
    clear_chroma_store()
    process_pdfs("data/")
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")

if __name__ == "__main__":
    main()