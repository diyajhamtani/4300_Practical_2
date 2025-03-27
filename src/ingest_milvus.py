import os
import re
import numpy as np
import pymilvus
from sentence_transformers import SentenceTransformer
import fitz  
import argparse
import ollama  
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema, utility

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Using Embedding Model: {EMBEDDING_MODEL}")

VECTOR_DIM = int(os.getenv("VECTOR_DIM", 384))
PREPROCESSING = os.getenv("PREPROCESSING", True)
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "COSINE"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))

# Available embedding models
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}

# Initialize the Milvus client
milvus_client = pymilvus.connections.connect("default", host="localhost", port="19530")

def create_milvus_collection():

    # if utility.has_collection(COLLECTION_NAME):
    #     print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it...")
    #     utility.drop_collection(COLLECTION_NAME)


    # Define the fields for the collection
    fields = [
       
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-increment primary key
        
        # Text data field
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        
        # Embedding vector field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]
    
    # Create a schema for the collection
    schema = CollectionSchema(fields, description="Embeddings for documents")

    # Create the collection
    collection = Collection(COLLECTION_NAME, schema)
    print("Milvus collection created successfully.")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:
    if "sentence-transformers" in model or "instructor" in model:
        transformer = SentenceTransformer(model)
        return transformer.encode(text).tolist()
    else:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]

def store_embedding(file: str, page: str, chunk: str, embedding: list, text: str):
    # Store the embedding and associated metadata in Milvus
    collection = Collection(COLLECTION_NAME)
    data = [
        [text],  # Text data
        [embedding],  # Embedding vector
    ]
    collection.insert(data)

    collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": DISTANCE_METRIC,
        "params": {"nlist": 128}
    }
)
    print(f"Stored embedding for: {chunk}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
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
                chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                        text=chunk,
                    )
            print(f" -----> Processed {file_name}")

def query_milvus(query_text: str):
    # Prepare the query
    embedding = get_embedding(query_text)
    
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # Use KNN search to retrieve the closest vectors
    search_params = {
        "metric_type": DISTANCE_METRIC,
        "params": {"nprobe": 10},  # Number of probes (increase for more accurate search)
    }
    search_result = collection.search(
        data=[embedding],  # Query vector
        anns_field="embedding",  # The field to search in
        param=search_params,
        limit=5,  # Return top 5 most similar vectors
        expr=None,  # No expression filter
        output_fields=["text"]
    )
    
    for result in search_result[0]:
        print(f"Text: {result.entity.text} \n Distance: {result.distance}\n")

def main():

    # Create Milvus collection if it doesn't exist
    create_milvus_collection()

    # Process PDFs and store embeddings in Milvus
    process_pdfs(os.path.join("data"))
    print("\n---Done processing PDFs---\n")

    # Query Milvus to find similar documents
    query_milvus("What is the capital of France?")

if __name__ == "__main__":
    main()
