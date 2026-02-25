"""Vector store using ChromaDB."""
import chromadb
from pathlib import Path

from tqdm import tqdm

import config
from src.chunking import Chunk
from src.embedding import embed_texts, embed_query


_chroma_client = None
_chroma_collection = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB persistent client (cached)."""
    global _chroma_client
    if _chroma_client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the medical reports collection (cached)."""
    global _chroma_collection
    if _chroma_collection is None:
        _chroma_collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


def index_chunks(chunks: list[Chunk], batch_size: int = 50):
    """
    Index chunks into ChromaDB with Gemini embeddings.

    Args:
        chunks: List of Chunk objects to index
        batch_size: Number of chunks to process at once
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if already indexed
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} documents. Skipping indexing.")
        print("Delete the chroma_db folder to re-index.")
        return

    num_batches = (len(chunks) + batch_size - 1) // batch_size
    pbar = tqdm(total=len(chunks), desc="Indexing chunks", unit="chunk")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.metadata for c in batch]

        # Generate embeddings
        embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        pbar.update(len(batch))

    pbar.close()

    print(f"Indexing complete. Total documents: {collection.count()}")


def search(query_text: str, n_results: int = None) -> dict:
    """
    Search for relevant chunks.

    Args:
        query_text: The search query
        n_results: Number of results to return

    Returns:
        ChromaDB query results dict with 'ids', 'documents', 'metadatas', 'distances'
    """
    if n_results is None:
        n_results = config.TOP_K

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Embed the query
    query_embedding = embed_query(query_text)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results
