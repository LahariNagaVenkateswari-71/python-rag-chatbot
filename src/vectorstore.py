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
        #EMbediing model used for converting text to vectors
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        #FAISS vector index
        #Initially empty vectors are added
        self.index = None

        ''' Stores chunk metadata
            Example:
            {
                "text": "...",
                "source": "python.pdf"
            }
         '''
        self.metadata = []

        #Load sentence transformer embedding model
        #used for text embeddings
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")


    def build_store(self, documents: List[Any]):

        # Create embedding Pipeline object
        # Handle chunking of documents
        emb_pipe = EmbeddingPipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        #Split documents in to chunks in the form of document structure
        #for better retrieval performance
        chunks = emb_pipe.chunk_documents(documents)

        # Store Cleaned chunk texts
        texts = []

        for chunk in chunks:
            #remove extra spaces or newline
            text = chunk.page_content.strip()

            if text:
                texts.append(text)

        print(f"[INFO] Total text chunks: {len(texts)}")

        #convert text chunk in to embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        #convert embeddings in to float32 format, required for FAISS
        embeddings = np.array(embeddings).astype("float32")
        print(f"[INFO] Embedding shape: {embeddings.shape}")

        #embedding vector dimention
        dim = embeddings.shape[1]

        # Create FAISS index and using L2 similarity search
        self.index = faiss.IndexFlatL2(dim)
        #add embeddings in to vector database
        self.index.add(embeddings)
        
        #store metadata for retrieval
        #used for multi-PDF reasoning
        self.metadata = []
        for chunk in chunks:

            text = chunk.page_content.strip()
            if text:

                #get source PDF name  and if source missing -> "unknown PDF"
                source = chunk.metadata.get(
                    "source",
                    "Unknown PDF")
                #store chunk text + source info
                self.metadata.append({

                    #Actual chunk content
                    "text": text,
                    #PDF source
                    "source": source    
                })
        #saving FAISS index + metadata locally
        self.save()
        print(f"[INFO] Added {len(texts)} vectors to FAISS index.")


    def save(self):
        faiss.write_index(
            self.index,
            os.path.join(self.persist_dir, "faiss.index"))
        with open(
            os.path.join(self.persist_dir, "metadata.pkl"), 
            "wb") as f:

            pickle.dump(self.metadata, f)


    def load(self):
        self.index = faiss.read_index(
            os.path.join(self.persist_dir, "faiss.index"))
        with open(
            os.path.join(self.persist_dir, "metadata.pkl"),
            "rb") as f:
            self.metadata = pickle.load(f)


    def query(self, query_text: str, top_k: int = 5):

        #converting user query in to embedding vector
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True).astype("float32")
        
        """ Search FAISS index
            top_k 2 retrieves extra results
            because we later re-rank using keyword matching - multi PDF reasoning
        """
        D, I = self.index.search(query_embedding, top_k * 2)

        #Store sementic and keyword results
        semantic_results = []

        #split hybrid query in to seperate words used for keyword matching
        query_words = set(query_text.lower().split())

        for idx in I[0]:

            item = self.metadata[idx]

            #extract chunk text
            text = item["text"]

            #Extract PDF source name
            source = item["source"]

            #split chunk text into words
            text_words = set(text.lower().split())

            #Calculate keyword overlap score
            #More common words = higher score
            keyword_score = len(
                query_words.intersection(text_words)
            )

            #store result with metadata
            semantic_results.append({
                "text": text,
                "source": source,
                "keyword_score": keyword_score
            })

        #Re-rank results using keyword score
        #Highest keyword matches come first
        semantic_results = sorted(
            semantic_results,
            key=lambda x: x["keyword_score"],
            reverse=True)
        
        #top hybrid search result
        final_results = semantic_results[:top_k]

        return final_results