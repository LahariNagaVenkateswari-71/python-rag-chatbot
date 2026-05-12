import os
import faiss
import pickle
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")


    def build_store(self, documents: List[Any]):

        emb_pipe = EmbeddingPipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)

        texts = []

        for chunk in chunks:

            text = chunk.page_content.strip()

            if text:
                texts.append(text)

        print(f"[INFO] Total text chunks: {len(texts)}")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = np.array(embeddings).astype("float32")

        print(f"[INFO] Embedding shape: {embeddings.shape}")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        self.metadata = texts

        self.save()

        print(f"[INFO] Added {len(texts)} vectors to FAISS index.")


    def save(self):

        faiss.write_index(
            self.index,
            os.path.join(self.persist_dir, "faiss.index")
        )

        with open(
            os.path.join(self.persist_dir, "metadata.pkl"),
            "wb"
        ) as f:

            pickle.dump(self.metadata, f)


    def load(self):

        self.index = faiss.read_index(
            os.path.join(self.persist_dir, "faiss.index")
        )

        with open(
            os.path.join(self.persist_dir, "metadata.pkl"),
            "rb"
        ) as f:

            self.metadata = pickle.load(f)


    def query(self, query_text: str, top_k: int = 5):

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True
        ).astype("float32")

        D, I = self.index.search(query_embedding, top_k)

        results = []

        for idx in I[0]:

            results.append(self.metadata[idx])

        return results