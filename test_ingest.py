import sqlite3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the same embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persistent DB
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 1. Check total chunks stored
collection = db._collection
print(f"Total chunks stored in DB: {collection.count()}")

# 2. Inspect raw data of top 2 chunks
results = collection.get(limit=2, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- CHUNK {i+1} ---")
    print(f"Source File: {meta.get('source')}")
    print(f"Text Snippet: {doc[:150]}...")