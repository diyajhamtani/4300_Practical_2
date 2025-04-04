import os
import psutil
import subprocess
import time
import pandas as pd
import search_chroma
import search_redis
import search_milvus
import chromadb

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
    "nomic": ["nomic-embed-text", 768]
}

LLM_MODELS = {
    "mistral": "mistral:latest",
    "llama2": "llama2",
}

DATABASES = ["redis", "chroma", "milvus"]

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # convert to mb

def get_memory_difference(starting_memory, label=""):
    ending_memory = get_memory()
    difference = ending_memory - starting_memory
    print(f"[{label}] Memory Difference: {difference} MB")
    return difference

def use_redis(embedding_model, llm_model, chunk_size, ingesting_time, ingesting_memory):
    all_rows = []
    
    for query in queries:
        try:
            start_time = time.time()
            start_memory = get_memory()
            context_results = search_redis.search_embeddings(query, embedding_model)
            response = search_redis.generate_rag_response(query, context_results, embedding_model)
            elapsed_time = time.time() - start_time
            memory = get_memory_difference(start_memory, f"Redis, {embedding_model}, {llm_model}")

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print(f"Memory: {memory:.2f} MB")

            all_rows.append(["redis", embedding_model, llm_model, query, elapsed_time, memory, response, ingesting_time, ingesting_memory, chunk_size])
        except Exception as e:
            print(f"Error processing query '{query}' in Redis: {e}")
            all_rows.append(["redis", embedding_model, llm_model, query, None, None, "ERROR", ingesting_time, ingesting_memory, chunk_size])

    return all_rows

def use_chroma(embedding_model, llm_model, chunk_size, ingesting_time, ingesting_memory):
    all_rows = []

    for query in queries:
        try:
            start_time = time.time()
            start_memory = get_memory()
            context_results = search_chroma.search_embeddings(query)
            response = search_chroma.generate_rag_response(query, context_results, llm_model)
            elapsed_time = time.time() - start_time
            memory = get_memory_difference(start_memory, f"Chroma, {embedding_model}, {llm_model}")

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print(f"Memory: {memory:.2f} MB")

            all_rows.append(["chroma", embedding_model, llm_model, query, elapsed_time, memory, response, ingesting_time, ingesting_memory, chunk_size])
        except Exception as e:
            print(f"Error processing query '{query}' in Chroma: {e}")
            all_rows.append(["chroma", embedding_model, llm_model, query, None, None, "ERROR", ingesting_time, ingesting_memory, chunk_size])

    return all_rows

def use_milvus(embedding_model, llm_model, chunk_size, ingesting_time, ingesting_memory):
    all_rows = []

    for query in queries:
        try:
            start_time = time.time()
            start_memory = get_memory()
            context_results = search_milvus.search_embeddings(query, embedding_model)
            response = search_milvus.generate_rag_response(query, context_results)
            elapsed_time = time.time() - start_time
            memory = get_memory_difference(start_memory, f"Milvus, {embedding_model}, {llm_model}")

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print(f"Memory: {memory:.2f} MB")

            all_rows.append(["milvus", embedding_model, llm_model, query, elapsed_time, memory, response, ingesting_time, ingesting_memory, chunk_size])
        except Exception as e:
            print(f"Error processing query '{query}' in Milvus: {e}")
            all_rows.append(["milvus", embedding_model, llm_model, query, None, None, "ERROR", ingesting_time, ingesting_memory, chunk_size])

    return all_rows


def main():
    os.environ["COLLECTION_NAME"] = "embedding_index"

    all_results = []
    errors = []
    preprocessing = True

    try:
        for embed_name, (embed_model, vector_dim) in EMBEDDING_MODELS.items():
            preprocessing = not preprocessing
            os.environ["EMBEDDING_MODEL"] = embed_model
            os.environ["VECTOR_DIM"] = str(vector_dim)
            os.environ["PREPROCESSING"] = str(preprocessing).lower()
            ingesting_time = {}
            ingesting_memory = {}

            # ingest data for each database based on embedding
            for database in DATABASES:
                start_time = time.time()
                start_ingesting_memory = get_memory()
                subprocess.run(["python", os.path.join("src", f"ingest_{database}.py")], check=True)
                ingesting_time[database] = time.time() - start_time
                ingesting_memory[database] = get_memory_difference(start_ingesting_memory)

            # test different llm models for each embedding
            for llm_name, llm_model in LLM_MODELS.items():
                os.environ["LLM_MODEL"] = llm_model

                print(f"\nRunning with Embedding Model: {embed_name} ({embed_model}) and LLM: {llm_name} ({llm_model})")

                # for minilm mistral redis combination, try different chunk sizes
                if embed_name == "minilm" and llm_model == "mistral:latest":  
                    chunk_sizes = [300, 1000]  
                    for chunk_size in chunk_sizes:
                        os.environ["CHUNK_SIZE"] = str(chunk_size)
                        start_time = time.time()
                        start_ingesting_memory = get_memory()

                        # reingest redis based on each chunk size
                        subprocess.run(["python", os.path.join("src", f"ingest_redis.py")], check=True)
                        ingesting_time["redis"] = time.time() - start_time
                        ingesting_memory["redis"] = get_memory_difference(start_ingesting_memory)

                        print(f"Running with Chunk Size: {chunk_size}")

                        # use redis for each chunk size
                        all_results.extend(use_redis(embed_model, llm_model, chunk_size, ingesting_time["redis"], ingesting_memory["redis"]))
                    
                    # set chunk size back to 300 for the other databases
                    chunk_size = 300
                    os.environ["CHUNK_SIZE"] = str(chunk_size)
                    all_results.extend(use_chroma(embed_model, llm_model, chunk_size, ingesting_time["chroma"], ingesting_memory["chroma"]))
                    all_results.extend(use_milvus(embed_model, llm_model, chunk_size, ingesting_time["milvus"], ingesting_memory["milvus"]))
               
                # for every other combination to test
                else:
                    chunk_size = 300
                    os.environ["CHUNK_SIZE"] = str(chunk_size)

                    # test each database for every embedding llm model combination
                    all_results.extend(use_chroma(embed_model, llm_model, chunk_size, ingesting_time["chroma"], ingesting_memory["chroma"]))
                    all_results.extend(use_milvus(embed_model, llm_model, chunk_size, ingesting_time["milvus"], ingesting_memory["milvus"]))
                    all_results.extend(use_redis(embed_model, llm_model, chunk_size, ingesting_time["redis"], ingesting_memory["redis"]))

                # save intermediate results in case all else fails
                df = pd.DataFrame(all_results, columns=["Database", "Embedding Model", "LLM Model", "Query", "Elapsed Time", "Memory", "Response", "ingesting_time", "ingesting_memory", "Chunk Size"])
                df.to_excel("intermediate.xlsx", index=False)
                print(f"Intermediate results saved to intermediate.xlsx")
            
            embed_num = list(EMBEDDING_MODELS.keys()).index(embed_name)
            os.environ["COLLECTION_NAME"] = f"embedding_index_{embed_num}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        errors.append(e)

    finally:
        
        for error in errors:
            print(error)

        # get final results xlsx
        df = pd.DataFrame(all_results, columns=["Database", "Embedding Model", "LLM Model", "Query", "Elapsed Time", "Memory", "Response", "ingesting_time", "ingesting_memory", "Chunk Size"])
        df.to_excel("results.xlsx", index=False)
        print("Results saved to results.xlsx")

if __name__ == "__main__":
    main()