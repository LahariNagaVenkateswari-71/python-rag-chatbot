import os
from src.load_data import load_all_documents
from src.vectorstore import FaissVectorStore

if __name__ == "__main__":

    store = FaissVectorStore("faiss_store")

    faiss_path = "faiss_store/faiss.index"
    meta_path = "faiss_store/metadata.pkl"

    # Build vector DB only first time
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):

        print("[INFO] Building vector store...")

        docs = load_all_documents("PDFs")

        store.build_store(docs)

    else:

        print("[INFO] Vector store already exists.")