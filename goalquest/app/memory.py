"""Long-term user memory in a vector store (Chroma locally; swap for Pinecone in prod).
Stores reflections, 'why' statements, and what motivated the user before, so the
Coach and Planner can retrieve them later."""
from __future__ import annotations

import logging
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

try:
    from chromadb import EmbeddingFunction

    class OfflineEmbedding(EmbeddingFunction):
        """Deterministic local embedding avoiding external network model downloads."""

        def __init__(self, dim: int = 32):
            self.dim = dim

        def name(self) -> str:
            return "offline_embedding"

        def __call__(self, input: Any) -> list[list[float]]:
            results = []
            for text in input:
                vec = [0.0] * self.dim
                for w in str(text).lower().split():
                    idx = abs(hash(w)) % self.dim
                    vec[idx] += 1.0
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
                results.append(vec)
            return results
except ImportError:
    OfflineEmbedding = None  # type: ignore


class MemoryStore:
    def __init__(self, path: str | None = None):
        self._fallback: list[dict] = []
        self.col = None
        try:
            import chromadb
            client = chromadb.PersistentClient(path=path or settings.chroma_path)
            ef = OfflineEmbedding() if OfflineEmbedding else None
            self.col = client.get_or_create_collection("user_memory", embedding_function=ef)
        except Exception as e:
            logger.warning("ChromaDB initialization failed, using in-memory store: %s", e)

    def add(self, user_id: int, text: str, kind: str, doc_id: str) -> None:
        if self.col is not None:
            try:
                self.col.upsert(ids=[f"{user_id}:{doc_id}"], documents=[text],
                                metadatas=[{"user_id": user_id, "kind": kind}])
                return
            except Exception as e:
                logger.warning("ChromaDB upsert failed, recording to in-memory fallback: %s", e)
        self._fallback.append({"id": f"{user_id}:{doc_id}", "user_id": user_id, "text": text, "kind": kind})

    def search(self, user_id: int, query: str, k: int = 3, kind: str | None = None) -> list[str]:
        if self.col is not None:
            try:
                where = {"user_id": user_id} if kind is None else {"$and": [{"user_id": user_id}, {"kind": kind}]}
                res = self.col.query(query_texts=[query], n_results=k, where=where)
                if res and res.get("documents") and res["documents"][0]:
                    return res["documents"][0]
            except Exception as e:
                logger.warning("ChromaDB query failed, using in-memory fallback: %s", e)
        # Fallback substring / keyword search
        q_words = [w for w in query.lower().split() if len(w) > 2]
        matches = [
            m["text"] for m in self._fallback
            if m["user_id"] == user_id and (kind is None or m["kind"] == kind)
            and (any(w in m["text"].lower() for w in q_words) if q_words else True)
        ]
        return matches[:k]
