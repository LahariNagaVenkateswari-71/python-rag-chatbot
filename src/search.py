import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self,persist_dir: str = "faiss_store",embedding_model: str = "all-MiniLM-L6-v2",llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # path to save FAISS vector index and metadata file
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        #check whether vector database exsistz
        if not (
            os.path.exists(faiss_path)
            and 
            os.path.exists(meta_path)
        ):
            from src.load_data import load_all_documents

            #Load all pdf documents
            docs = load_all_documents("PDFs")

            #build FAISS vector database
            self.vectorstore.build_store(docs)
        else:

            #load exsisting vector database
            self.vectorstore.load()

        # Groq API Key
        groq_api_key = os.getenv("GROQ_API_KEY")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq Model Loaded: {llm_model}")



    def search_and_summarize(self, query: str, chat_history=None, top_k: int = 2) -> str:

        #Retrive relevent chunks from vector store. Uses hybrid search
        results = self.vectorstore.query(query, top_k=top_k)

        #store retrieved chunk texts
        contexts = []

        #store unique pdf sources and set() automatically remove duplicates
        sources = set()

        for item in results:
            contexts.append(item["text"]) #add retrieved chunk text 
            sources.add(item["source"]) # Add source PDF name

        # combine all contexts in to one context
        # used for LLM prompt
        context = "\n\n".join(contexts)

        #source names in to string
        source_text = "\n".join(sources)
        history = ""        # store previous conversation history

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

        Sources:
        {source_text}
        """

        #send promp to LLM
        response = self.llm.invoke(prompt)
        return response.content