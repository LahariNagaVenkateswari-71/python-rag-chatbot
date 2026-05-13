import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self,persist_dir: str = "faiss_store",embedding_model: str = "all-MiniLM-L6-v2",llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load vector DB
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.load_data import load_all_documents
            docs = load_all_documents("PDFs")
            self.vectorstore.build_store(docs)
        else:
            self.vectorstore.load()

        # Groq API Key
        groq_api_key = os.getenv("GROQ_API_KEY")

        # LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq Model Loaded: {llm_model}")

    def search_and_summarize(self, query: str, chat_history=None, top_k: int = 2) -> str:

        results = self.vectorstore.query(query, top_k=top_k)
        texts = results
        context = "\n\n".join(texts)
        history = ""

        if chat_history:
            for msg in chat_history[-4:]:
                history += f"{msg['role']}: {msg['content']}\n"

        prompt = f"""
        You are a friendly AI Python assistant.

        Use previous conversation history to respond naturally.

        If the user asks Python-related questions,
        use the provided documentation context.

        Conversation History:
        {history}

        Documentation Context:
        {context}

        User Question:
        {query}
        """

        response = self.llm.invoke(prompt)
        return response.content