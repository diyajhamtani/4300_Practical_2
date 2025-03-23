import os
import subprocess
import time
import pandas as pd
import search_chroma
import search_redis
import search_milvus

queries = [
    "What is binary search?",
    "Add 23 to the AVL tree with elements 30, 25, 35, 20 (30 is at the root). what imbalance case is created when inserting 23?",
    "Succinctly describe the four components of ACID compliant transactions.",
    "Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre?",
    "When was SQL originally released?"
]

EMBEDDING_MODELS = {
    "minilm": ["sentence-transformers/all-MiniLM-L6-v2", 384],
    "mpnet": ["sentence-transformers/all-mpnet-base-v2", 768],
    "instructor": ["hkunlp/instructor-xl", 768]
}

LLM_MODELS = {
    "mistral": "mistral:latest",
    "llama2": "llama2",
    "deepseek": "deepseek-r1"
}

DATABASES = {
    'redis': 'redis',
    'chroma': 'chroma',
    'milvus': 'milvus'
}

def use_redis(embedding_model, llm_model):
    start_time = time.time()
    subprocess.run(["python", os.path.join("src", "ingest_redis.py")], check=True)
    ingesting_time = time.time() - start_time
    all_rows = []
    
    for query in queries:
        try:
            start_time = time.time()
            context_results = search_redis.search_embeddings(query, embedding_model)
            response = search_redis.generate_rag_response(query, context_results, embedding_model)
            elapsed_time = time.time() - start_time

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            all_rows.append(["redis", embedding_model, llm_model, query, elapsed_time, response, ingesting_time])
        except Exception as e:
            print(f"Error processing query '{query}' in Redis: {e}")
            all_rows.append(["redis", embedding_model, llm_model, query, None, "ERROR", ingesting_time])

    return all_rows


def use_chroma(embedding_model, llm_model):
    start_time = time.time()
    subprocess.run(["python", os.path.join("src", "ingest_chroma.py")], check=True)
    ingesting_time = time.time() - start_time
    all_rows = []

    for query in queries:
        try:
            start_time = time.time()
            context_results = search_chroma.search_embeddings(query)
            response = search_chroma.generate_rag_response(query, context_results, llm_model)
            elapsed_time = time.time() - start_time

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            all_rows.append(["chroma", embedding_model, llm_model, query, elapsed_time, response, ingesting_time])
        except Exception as e:
            print(f"Error processing query '{query}' in Chroma: {e}")
            all_rows.append(["chroma", embedding_model, llm_model, query, None, "ERROR", ingesting_time])

    return all_rows

def use_milvus(embedding_model, llm_model):
    start_time = time.time()
    subprocess.run(["python", os.path.join("src", "ingest_milvus.py")], check=True)
    ingesting_time = time.time() - start_time
    all_rows = []

    for query in queries:
        try:
            start_time = time.time()
            context_results = search_milvus.search_embeddings(query, embedding_model)
            response = search_milvus.generate_rag_response(query, context_results)
            elapsed_time = time.time() - start_time

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            all_rows.append(["milvus", embedding_model, llm_model, query, elapsed_time, response, ingesting_time])
        except Exception as e:
            print(f"Error processing query '{query}' in Milvus: {e}")
            all_rows.append(["milvus", embedding_model, llm_model, query, None, "ERROR", ingesting_time])

    return all_rows


def main():
    all_results = []

    try:
        for embed_name, (embed_model, vector_dim) in EMBEDDING_MODELS.items():
            for llm_name, llm_model in LLM_MODELS.items():
                os.environ["EMBEDDING_MODEL"] = embed_model
                os.environ["LLM_MODEL"] = llm_model
                os.environ["VECTOR_DIM"] = str(vector_dim)

                print(f"\nRunning with Embedding Model: {embed_name} ({embed_model}) and LLM: {llm_name} ({llm_model})")

                chroma_results = use_chroma(embed_model, llm_model)
                redis_results = use_redis(embed_model, llm_model)

                all_results.extend(redis_results + chroma_results)

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        # Ensure CSV is always saved even if an error occurs
        df = pd.DataFrame(all_results, columns=["Database", "Embedding Model", "LLM Model", "Query", "Elapsed Time", "Response"])
        df.to_csv("results.csv", index=False)
        print("Results saved to results.csv")

if __name__ == "__main__":
    main()
