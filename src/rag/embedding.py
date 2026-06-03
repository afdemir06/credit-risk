import logging
import requests
import os
import chromadb
from chromadb.config import Settings
from src.config import POLICIES_DIR

logger=logging.getLogger(__name__)

OLLAMA_BASE=os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME="credit_policies"
CHROMA_DIR=POLICIES_DIR / "chroma_db"

def get_embeddings(texts: list):
    try:
        response=requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120
        )
        response.raise_for_status()
        result=response.json()
        return result["embeddings"]
    except requests.exceptions.ConnectionError:
        logger.error(f"Ollama not running at {OLLAMA_BASE}")
        raise
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def get_chroma_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

def get_or_create_collection():
    client=get_chroma_client()
    try:
        collection=client.get_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' loaded")
    except Exception:
        collection=client.create_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' created")
    return collection

def store_chunks(chunks: list, source: str):
    collection=get_or_create_collection()
    logger.info(f"Embedding {len(chunks)} chunks via Ollama ({EMBED_MODEL})")
    embeddings=get_embeddings(chunks)
    ids=[f"{source}_{i}" for i in range(len(chunks))]
    metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    logger.info(f"Stored {len(chunks)} chunks from '{source}'")
    return len(chunks)

def collection_count():
    collection=get_or_create_collection()
    return collection.count()
