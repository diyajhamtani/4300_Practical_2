import os
import subprocess

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-xl"
}
DATABASES = {
    'redis',
    'chroma'
}

print("Select an embedding model:")
for key, model in EMBEDDING_MODELS.items():
    print(f"{key}: {model}")
user_choice = input("Enter model key: ").strip().lower()
selected_model = EMBEDDING_MODELS.get(user_choice, EMBEDDING_MODELS["minilm"])
print(f"Using model: {selected_model}")

print("Select a database:")
for key, model in DATABASES.items():
    print(f"{key}: {model}")
user_choice = input("Enter model key: ").strip().lower()
selected_db = EMBEDDING_MODELS.get(user_choice, EMBEDDING_MODELS["minilm"])

os.environ["EMBEDDING_MODEL"] = selected_model

subprocess.run(["python", "src/ingest_{selected_db}.py"])
subprocess.run(["python", "src/search_{selected_db}.py"])
