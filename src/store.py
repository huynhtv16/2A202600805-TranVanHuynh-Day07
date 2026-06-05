from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            from chromadb.config import Settings

            client = chromadb.Client(Settings())
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for record in records:
            score = _dot(query_embedding, record.get("embedding", []))
            scored.append({
                "id": record.get("id"),
                "content": record.get("content"),
                "metadata": record.get("metadata", {}),
                "score": score,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        if self._use_chroma and self._collection is not None:
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embeddings: list[list[float]] = []

            for doc in docs:
                ids.append(doc.id)
                documents.append(doc.content)
                metadatas.append(doc.metadata or {})
                embeddings.append(self._embedding_fn(doc.content))
                self._store.append(self._make_record(doc))

            try:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except Exception:
                pass
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._use_chroma and self._collection is not None:
            try:
                query_embedding = self._embedding_fn(query)
                response = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                )
                results: list[dict[str, Any]] = []
                ids = response.get("ids", [[]])[0]
                documents = response.get("documents", [[]])[0]
                metadatas = response.get("metadatas", [[]])[0]
                distances = response.get("distances", [[]])[0]
                for idx in range(len(ids)):
                    results.append({
                        "id": ids[idx],
                        "content": documents[idx],
                        "metadata": metadatas[idx],
                        "score": distances[idx],
                    })
                return results[:top_k]
            except Exception:
                pass

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        filtered_records = [
            record
            for record in self._store
            if all(record.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        original_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record.get("id") != doc_id and record.get("metadata", {}).get("doc_id") != doc_id
        ]
        deleted = original_size - len(self._store)

        if deleted and self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass

        return deleted > 0
