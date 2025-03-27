import os
import numpy as np
import ollama
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

if "default" not in connections.list_connections():
    connections.connect("default", host="localhost", port="19530")
    
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 384))
INDEX_NAME = "embedding_collection"
DISTANCE_METRIC = "COSINE"
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Using LLM: {LLM_MODEL}")
print(f"Using Embedding Model: {EMBEDDING_MODEL}")


def get_embedding(text: str, model: str) -> list:
    if "sentence-transformers" in model or "instructor" in model:
        transformer = SentenceTransformer(model)
        return transformer.encode(text).tolist()
    else:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]


def search_embeddings(query, embedding_model, top_k=3):
    query_embedding = get_embedding(query, embedding_model)
    collection = Collection(INDEX_NAME)
    collection.load()

    search_params = {
        "metric_type": DISTANCE_METRIC,
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    top_results = [
        {
            "text": result.entity.text,
            "similarity": result.distance,
        }
        for result in results[0]
    ]

    for result in top_results:
        print(f"---> Filw: {result['text']}\nSimilarity: {result['similarity']:.4f}\n")

    return top_results


def generate_rag_response(query, context_results, llm_model=LLM_MODEL):
    context_str = "\n".join([
        f"{res['text']}\n(Similarity: {res['similarity']:.4f})" for res in context_results
    ])

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=llm_model, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def interactive_search(embedding_model):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, embedding_model)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search(EMBEDDING_MODEL)
