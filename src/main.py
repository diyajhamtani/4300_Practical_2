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
selected_db = DATABASES.get(user_choice, "redis")  # Default to "redis" if input is invalid

# Set environment variable
os.environ["EMBEDDING_MODEL"] = selected_model

# Run scripts
# Run scripts
subprocess.run(["python", f"src/ingest_{selected_db}.py", "--embedding_model", selected_model])
subprocess.run(["python", f"src/search_{selected_db}.py", "--embedding_model", selected_model])
