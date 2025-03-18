import redis
import json
import os
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField


# Initialize Redis client
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Get LLM model from environment variable
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")

print(f"Using LLM: {LLM_MODEL}")

def get_embedding(text: str, model: str) -> list:
    if "sentence-transformers" in model or "instructor" in model:
        transformer = SentenceTransformer(model)
        return transformer.encode(text).tolist()
    else:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]


def search_embeddings(query, embedding_model, top_k=3):
    query_embedding = get_embedding(query, embedding_model)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, embedding_model):
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
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}]
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
        response = generate_rag_response(query, context_results, embedding_model)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Search with Redis")
    parser.add_argument("--embedding_model", type=str, required=True, help="Specify the embedding model")
    args = parser.parse_args()

    interactive_search(args.embedding_model)
