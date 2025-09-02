import chromadb
from chromadb.config import Settings
import os
import numpy as np

LOGS_DIR = 'multimodal_logs'
MEMORY_DB_DIR = os.path.join(LOGS_DIR, 'vector_db')

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=MEMORY_DB_DIR, settings=Settings(allow_reset=True))

# Create or get the collection
collection = client.get_or_create_collection("multimodal_memories")

# --- Memory Manager Functions ---
def add_memory(embedding, metadata):
    """Add a new memory (embedding + metadata) to the vector DB."""
    # ChromaDB expects lists, not numpy arrays
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[metadata.get('id', str(metadata.get('timestamp', 'unknown')))]
    )


def search_memories(query_embedding, top_k=5):
    """Search for the most relevant memories given a query embedding."""
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    # Return list of (metadata, distance)
    hits = []
    for md, dist in zip(results['metadatas'][0], results['distances'][0]):
        hits.append((md, dist))
    return hits

# Example usage (uncomment to test):
# add_memory(np.random.rand(512), {"type": "vision", "timestamp": 1234567890, "caption": "A cat on a sofa"})
# print(search_memories(np.random.rand(512))) 