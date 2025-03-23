import os
import psutil
import subprocess
import time
import pandas as pd
import search_chroma
import search_redis
import search_milvus
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    subprocess.run(["python", os.path.join("src", "ingest_redis.py")], check=True)
    return process_queries(search_redis.search_embeddings, search_redis.generate_rag_response, "redis", embedding_model, llm_model)

def use_chroma(embedding_model, llm_model):
    subprocess.run(["python", os.path.join("src", "ingest_chroma.py")], check=True)
    return process_queries(search_chroma.search_embeddings, search_chroma.generate_rag_response, "chroma", embedding_model, llm_model)

def use_milvus(embedding_model, llm_model):
    subprocess.run(["python", os.path.join("src", "ingest_milvus.py")], check=True)
    return process_queries(search_milvus.search_embeddings, search_milvus.generate_rag_response, "milvus", embedding_model, llm_model)

def process_queries(search_func, response_func, db_name, embedding_model, llm_model):
    all_rows = []
    
    def process_query(query):
        try:
            start_time = time.time()
            start_memory = get_memory()
            context_results = search_func(query, embedding_model[0])
            response = response_func(query, context_results, llm_model)
            elapsed_time = time.time() - start_time
            memory = get_memory_difference(start_memory, f"{db_name}, {embedding_model}, {llm_model}")
            logging.info(f"{db_name} | {query} | Time: {elapsed_time:.4f}s | Memory: {memory:.2f} MB")
            return [db_name, embedding_model[0], llm_model, query, elapsed_time, response]
        except Exception as e:
            logging.error(f"Error processing query '{query}' in {db_name}: {e}")
            return [db_name, embedding_model[0], llm_model, query, None, "ERROR"]
    
    with ThreadPoolExecutor() as executor:
        all_rows = list(executor.map(process_query, queries))
    
    return all_rows

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # convert to mb

def get_memory_difference(starting_memory, label=""):
    ending_memory = get_memory()
    difference = ending_memory - starting_memory
    logging.info(f"[{label}] Memory Difference: {difference} MB")
    return difference

def main():
    all_results = []
    
    try:
        for embed_name, embed_model in EMBEDDING_MODELS.items():
            for llm_name, llm_model in LLM_MODELS.items():
                os.environ["EMBEDDING_MODEL"] = embed_model[0]
                os.environ["LLM_MODEL"] = llm_model
                os.environ["VECTOR_DIM"] = str(embed_model[1])
                
                logging.info(f"\nRunning with Embedding Model: {embed_name} and LLM: {llm_name}")
                
                all_results.extend(use_chroma(embed_model, llm_model))
                all_results.extend(use_redis(embed_model, llm_model))
                all_results.extend(use_milvus(embed_model, llm_model))
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    
    finally:
        df = pd.DataFrame(all_results, columns=["Database", "Embedding Model", "LLM Model", "Query", "Elapsed Time", "Response"])
        df.to_csv("results.csv", index=False)
        logging.info("Results saved to results.csv")

if __name__ == "__main__":
    main()
