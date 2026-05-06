from src.load_data import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

if __name__ == "__main__":
    docs = load_all_documents("PDFs")
    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunks_vectors = EmbeddingPipeline().embed_chunks(chunks)
    store = FaissVectorStore("fiass_store")
    store.load()
    print(store.quary("who invented python?",top_k=2))

