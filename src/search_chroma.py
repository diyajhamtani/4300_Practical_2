import chromadb
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import ollama

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768

# Get embedding model from environment variable
CURRENT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
print(f"Using model: {CURRENT_MODEL}")

def get_embedding(text: str, model: str = CURRENT_MODEL) -> list:
    if "sentence-transformers" in model or "instructor" in model:
        transformer = SentenceTransformer(model)
        return transformer.encode(text).tolist()
    else:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]

def search_embeddings(query, top_k=3):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    top_results = [
        {
            "file": result["file"],
            "page": result["page"],
            "chunk": result["chunk"],
            "similarity": score
        }
        for result, score in zip(results["metadatas"][0], results["distances"][0])
    ]
    
    for result in top_results:
        print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")
    
    return top_results

def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    
    print(f"context_str: {context_str}")
    
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break
        
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
