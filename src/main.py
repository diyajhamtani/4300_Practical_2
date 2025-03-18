import os
import subprocess

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}
DATABASES = {
    'redis': 'redis',
    'chroma': 'chroma'
}
LLM_MODELS = {
    "mistral": "mistral:latest",
    "llama2": "llama2",
    "deepseek": "deepseek-r1"
}

# Select embedding model
print("Select an embedding model:")
for key, model in EMBEDDING_MODELS.items():
    print(f"{key}: {model}")
user_choice = input("Enter model key: ").strip().lower()
selected_model = EMBEDDING_MODELS.get(user_choice, EMBEDDING_MODELS["minilm"])
print(f"Using model: {selected_model}")

# Select database
print("Select a database:")
for key in DATABASES:
    print(f"{key}")
user_choice = input("Enter database key: ").strip().lower()
selected_db = DATABASES.get(user_choice, DATABASES["redis"])  # Default to "redis" if input is invalid

# Select LLM
print("Select a database:")
for key in LLM_MODELS:
    print(f"{key}")
user_choice = input("Enter llm key: ").strip().lower()
selected_llm = LLM_MODELS.get(user_choice, LLM_MODELS["mistral"])  # Default to "mistral:latest" if input is invalid

# Set environment variable
os.environ["EMBEDDING_MODEL"] = selected_model
os.environ["LLM_MODEL"] = selected_llm

# Run scripts
subprocess.run(["python", os.path.join("src", f"ingest_{selected_db}.py"), "--embedding_model", selected_model])
subprocess.run(["python", os.path.join("src", f"search_{selected_db}.py"), "--embedding_model", selected_model])
