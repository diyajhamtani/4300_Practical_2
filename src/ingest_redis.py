import redis
import json
import numpy as np
import os
import fitz
import ollama
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import re

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Using Embedding Model: {EMBEDDING_MODEL}")

PREPROCESSING = os.getenv("PREPROCESSING", TRUE)
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 384))
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index():
    """Creates a new HNSW index in Redis."""
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def get_embedding(text: str) -> list:
    """Generate an embedding for the given text."""
    try:
        if "sentence-transformers" in EMBEDDING_MODEL or "instructor" in EMBEDDING_MODEL:
            transformer = SentenceTransformer(EMBEDDING_MODEL)
            return transformer.encode(text).tolist()
        else:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Embedding generation error: {e}")

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """Store text embeddings in Redis."""
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
    print(f"Stored embedding for: {chunk}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
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
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                # Preprocess text before chunking
                if PREPROCESSING:
                    text = preprocess_text(text)
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding
                    )
            print(f" -----> Processed {file_name}")

def query_redis(query_text: str):
    """Searches Redis for the most relevant embeddings."""
    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "vector_distance")
            .dialect(2)
        )

        embedding = get_embedding(query_text)
        res = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )

        if not res.docs:
            print("No relevant results found.")
            return

        print("\nSearch Results:")
        for doc in res.docs:
            print(f" {doc.id} \n ----> Similarity Score: {doc.vector_distance}\n")

    except Exception as e:
        raise RuntimeError(f"Redis query error: {e}")

def main():
    clear_redis_store()
    create_hnsw_index()
    process_pdfs(os.path.join("data"))
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()
