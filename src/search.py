import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self,persist_dir: str = "faiss_store",embedding_model: str = "all-MiniLM-L6-v2",llm_model: str = "llama3-8b-8192"):
        self.vectorstore = FaissVectorStore()
        # Load vector DB
        faiss_path = os.path.join(persist_dir,"faiss.index")
        meta_path = os.path.join(persist_dir,"metadata.pkl")
        if (os.path.exists(faiss_path)and os.path.exists(meta_path)):
            self.vectorstore.load()

        # Groq API Key
        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        # LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq Model Loaded: "
            f"{llm_model}"
        )

    def search_and_summarize(self,query: str,top_k: int = 3) -> str:
        results = self.vectorstore.search(query,top_k=top_k)
        context = "\n\n".join(results)
        if not context:
            return "No relevant documents found."
        prompt = f"""
        Answer the question using the context below.

        Question:
        {query}

        Context:
        {context}
        """
        response = self.llm.invoke(prompt)

        return response.content