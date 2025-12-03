"""
vector_store.py

Builds and queries a FAISS vector store using sentence-transformers
for embeddings.

Responsibilities:
- Compute embeddings for text chunks
- Build FAISS index
- Save/load index + metadata
- Run similarity search (top-k)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ChunkMetadata:
    chunk_id: int
    page_num: int


class FaissVectorStore:
    def __init__(
        self,
        index: faiss.Index,
        embeddings_dim: int,
        metadata: List[ChunkMetadata],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.index = index
        self.embeddings_dim = embeddings_dim
        self.metadata = metadata
        self.embedding_model_name = embedding_model_name
        self._model = SentenceTransformer(self.embedding_model_name)

    @classmethod
    def from_chunks(
        cls,
        chunks: List[Dict],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> Tuple["FaissVectorStore", np.ndarray]:
        """
        Build a new FAISS index from a list of chunk dicts:
        - chunks[i]["text"], chunks[i]["chunk_id"], chunks[i]["page_num"]

        Returns (vector_store, embeddings_array)
        """
        texts = [c["text"] for c in chunks]
        metadata = [
            ChunkMetadata(chunk_id=c["chunk_id"], page_num=c["page_num"])
            for c in chunks
        ]

        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine

        index.add(embeddings)

        store = cls(
            index=index,
            embeddings_dim=dim,
            metadata=metadata,
            embedding_model_name=embedding_model_name,
        )

        return store, embeddings

    def save(self, output_dir: str | Path) -> None:
        """
        Save index + metadata to a directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) save FAISS index
        faiss_path = output_dir / "index.faiss"
        faiss.write_index(self.index, str(faiss_path))

        # 2) save metadata + config
        meta = {
            "embeddings_dim": self.embeddings_dim,
            "embedding_model_name": self.embedding_model_name,
            "metadata": [asdict(m) for m in self.metadata],
        }
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str | Path) -> "FaissVectorStore":
        """
        Load an existing FAISS index + metadata from a directory.
        """
        output_dir = Path(output_dir)

        faiss_path = output_dir / "index.faiss"
        meta_path = output_dir / "metadata.json"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS index or metadata in {output_dir}"
            )

        index = faiss.read_index(str(faiss_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        embeddings_dim = meta["embeddings_dim"]
        embedding_model_name = meta.get(
            "embedding_model_name", DEFAULT_EMBEDDING_MODEL
        )
        metadata = [ChunkMetadata(**m) for m in meta["metadata"]]

        store = cls(
            index=index,
            embeddings_dim=embeddings_dim,
            metadata=metadata,
            embedding_model_name=embedding_model_name,
        )
        return store

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string and normalize it.
        """
        vec = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        return vec

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search and return top-k metadata + scores.

        Returns:
        [
            {
                "score": float,
                "chunk_metadata": ChunkMetadata,
                "index": int
            },
            ...
        ]
        """
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "chunk_metadata": self.metadata[idx],
                    "index": int(idx),
                }
            )
        return results
