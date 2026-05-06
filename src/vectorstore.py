import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.folder = "faiss_store"
        os.makedirs(self.folder, exist_ok=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_store(self, documents):
        emb = EmbeddingPipeline()
        chunks = emb.chunk_documents(documents)
        vectors = emb.embed_chunks(chunks)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(vectors).astype("float32"))

        for chunk in chunks:
            self.metadata.append(chunk.page_content)

        self.save()

    def save(self):
        faiss.write_index(self.index,f"{self.folder}/faiss.index")

        with open(f"{self.folder}/metadata.pkl","wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(f"{self.folder}/faiss.index")
        with open(f"{self.folder}/metadata.pkl","rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query, top_k=3):
        query_vector = self.model.encode([query])
        D, I = self.index.search(
            np.array(query_vector).astype("float32"),
            top_k
        )
        results = []
        for i in I[0]:
            results.append(self.metadata[i])
        return results